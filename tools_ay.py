import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# =========================
# Helpers
# =========================

def to_tensor(x, device, requires_grad: bool = False, dtype=torch.float32):
    """Safe wrapping to tensor without copy-construct warnings."""
    if torch.is_tensor(x):
        t = x.detach().clone().to(device=device, dtype=dtype)
        t.requires_grad_(requires_grad)
        return t
    t = torch.as_tensor(x, dtype=dtype, device=device)
    t.requires_grad_(requires_grad)
    return t


def unit_norm_atoms_(phi, eps=1e-12):
    """Project each atom (K,L,P) to unit l2 norm."""
    K = phi.shape[0]
    flat = phi.view(K, -1)
    norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min_(eps)
    phi.copy_((flat / norms).view_as(phi))
    return phi


# =========================
# Initialization
# =========================

def setInitialValues(X, K, M, L, *, device=None):
    """
    X: (S, N, P)
    returns:
      phi: (K, L, P)
      Z:   (S, K, N-L+1)
      A:   (S, K, M)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = to_tensor(X, device)
    S, N, P = X.shape

    phi = torch.zeros((K, L, P), device=device)
    for k in range(K):
        start = torch.randint(0, N - L + 1, (1,), device=device).item()
        atom = X[0, start:start + L, :]  # (L,P)
        nrm = torch.linalg.vector_norm(atom)
        if nrm > 1e-12:
            phi[k] = atom / nrm
        else:
            phi[k] = torch.randn(L, P, device=device)
            phi[k] /= torch.linalg.vector_norm(phi[k]) + 1e-12

    Z = torch.zeros((S, K, N - L + 1), device=device)
    A = torch.zeros((S, K, M), device=device)
    return phi, Z, A


# =========================
# Batched CSC (ℓ0, no overlap)
# =========================

def ensure_same_device_dtype(*tensors, ref=None, dtype=torch.float32):
    """
    Move all tensors to the same device & dtype.
    If ref is given, use ref.device; else use the first tensor's device, else cuda if available.
    Returns a tuple of tensors (moved).
    """
    # pick device
    if ref is not None:
        device = ref.device
    else:
        # find first tensor with .device, else default
        device = None
        for t in tensors:
            if torch.is_tensor(t):
                device = t.device
                break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moved = []
    for t in tensors:
        if torch.is_tensor(t):
            moved.append(t.detach().clone().to(device=device, dtype=dtype))
        else:
            moved.append(torch.as_tensor(t, dtype=dtype, device=device))
    return tuple(moved)

@torch.no_grad()
def CSC(X, Z, phi, lamb, max_iters=50, *, device=None):
    """
    X:   (S, N, P)
    Z:   (S, K, N-L+1)  [will be filled]
    phi: (K, L, P)
    lamb: threshold for correlation score
    returns: Z (same shape)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = to_tensor(X, device)
    Z = to_tensor(Z, device)
    phi = to_tensor(phi, device)

    S, N, P = X.shape
    K, L, P2 = phi.shape
    assert P2 == P, "phi last dim must match X channels"

    for s in range(S):
        residual = X[s].clone()  # (N,P)
        Z_s = Z[s]               # (K, N-L+1)

        it = 0
        while torch.linalg.vector_norm(residual) > 1e-6 and it < max_iters:
            # correlation per atom (multi-channel)
            # do conv as correlation: input (1,P,N), weight (1,P,L)
            inp = residual.permute(1, 0).unsqueeze(0)  # (1,P,N)
            best_k, best_t, best_val = None, None, 0.0

            for k in range(K):
                # cross-correlation with phi[k] (no flip to do correlation)
                w = phi[k].permute(1, 0).unsqueeze(0)  # (1,P,L)
                # valid = length N-L+1
                score = F.conv1d(inp, w, padding=0)    # (1,1,N-L+1)
                score = score.squeeze(0).squeeze(0)    # (N-L+1,)

                # pick max > lamb
                val, idx = torch.max(score, dim=0)
                if val.item() > lamb and val.item() > best_val:
                    best_val = val.item()
                    best_k = k
                    best_t = idx.item()

            if best_k is None:
                break  # no more activations

            # place activation with amplitude = correlation score
            Z_s[best_k, best_t] += best_val

            # subtract amplitude * atom from residual at [t:t+L]
            t0 = best_t
            residual[t0:t0 + L, :] -= best_val * phi[best_k, :, :]

            it += 1

        Z[s] = Z_s

    return Z


@torch.no_grad()
def CSC_wta_nms(
    X, Z, phi,
    lamb: float,
    max_outer_iters: int = 20,
    adaptive_lambda: bool = False,
):
    """
    Winner-takes-time CSC with global non-overlap across atoms.
    X:   (S, N, P)
    Z:   (S, K, N-L+1)  [updated & returned]
    phi: (K, L, P)
    """
    # ---- unify device & dtype (use phi's device if it’s CUDA) ----
    X, Z, phi = ensure_same_device_dtype(X, Z, phi, ref=phi)

    S, N, P = X.shape
    K, L, P2 = phi.shape
    assert P2 == P, "phi last dim must match X channels"

    # unit-normalize atoms (scale lives in Z)
    flat = phi.view(K, -1)
    norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min_(1e-12)
    phi = (flat / norms).view_as(phi).contiguous()

    residual = X.clone()  # (S,N,P)

    # conv weights: correlation for scoring, convolution for synthesis
    corr_weight  = phi.permute(0, 2, 1).contiguous()           # (K,P,L)
    synth_weight = phi.permute(2, 0, 1).flip(-1).contiguous()  # (P,K,L)

    T = N - L + 1

    for _ in range(max_outer_iters):
        # (S,P,N) -> conv1d with (K,P,L) -> (S,K,T)
        scores = F.conv1d(residual.permute(0, 2, 1), corr_weight, padding=0)

        # best atom per time (winner-takes-time across K)
        best_val, best_k = scores.max(dim=1)  # (S,T)

        # thresholding
        if adaptive_lambda:
            pos = best_val.clamp_min_(0.0)
            med = pos.median(dim=1, keepdim=True).values  # (S,1)
            lam_s = (0.75 * med).expand_as(best_val)      # tune factor if needed
            mask = best_val > lam_s
        else:
            mask = best_val > lamb

        if not mask.any():
            break

        # Non-maximum suppression along time (window L)
        best_val_   = best_val.view(S, 1, T)                     # (S,1,T)
        pooled      = F.max_pool1d(best_val_, kernel_size=L, stride=1)  # (S,1,T-L+1)
        pooled_full = F.pad(pooled, (0, L-1, 0, 0, 0, 0), value=float("-inf"))  # (S,1,T)
        keep        = (best_val_ >= pooled_full).squeeze(1) & mask            # (S,T)

        if not keep.any():
            break

        # Build ΔZ with the chosen (s, k*, t)
        dZ = torch.zeros_like(Z)               # (S,K,T)
        s_idx, t_idx = torch.where(keep)       # indices kept
        k_idx = best_k[s_idx, t_idx]
        amp   = best_val[s_idx, t_idx]
        dZ[s_idx, k_idx, t_idx] = amp

        # Single-shot synthesis and residual update
        contrib = F.conv1d(dZ, synth_weight, padding=L-1)  # (S,P,N)
        residual -= contrib.permute(0, 2, 1)               # -> (S,N,P)
        Z += dZ

        if torch.linalg.vector_norm(residual) < 1e-6:
            break

    return Z

@torch.no_grad()
def CSC_batched(
    X,
    Z,
    phi,
    lamb: float,
    max_outer_iters: int = 20,
    pick_mode: str = "nms",  # "nms" or "global"
):
    """
    Parallel greedy CSC with non-maximum suppression per (subject,atom) row.
    X:   (S, N, P)
    Z:   (S, K, N-L+1)
    phi: (K, L, P)
    lamb: threshold on correlation scores
    returns: Z (same device as phi/X)
    """
    # ---- Choose a common device & dtype ----
    # Prefer phi's device if available, else X's, else CPU
    if torch.is_tensor(phi) and phi.is_cuda:
        device = phi.device
    elif torch.is_tensor(X) and getattr(X, "is_cuda", False):
        device = X.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32

    # Move all to the same device/dtype (and clone to avoid warnings)
    def to_t(x):
        if torch.is_tensor(x):
            return x.detach().clone().to(device=device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=device)

    X = to_t(X)        # (S,N,P)
    Z = to_t(Z)        # (S,K,T)
    phi = to_t(phi)    # (K,L,P)

    S, N, P = X.shape
    K, L, P2 = phi.shape
    assert P2 == P, "phi last dim must match X channels"

    residual = X.clone()  # (S,N,P)

    # Correlation weight (no flip)
    corr_weight = phi.permute(0, 2, 1).contiguous()   # (K,P,L)
    # Synthesis weight (flip for true convolution)
    synth_weight = phi.permute(2, 0, 1).flip(-1).contiguous()  # (P,K,L)

    T = N - L + 1

    for _ in range(max_outer_iters):
        # residual (S,N,P) -> (S,P,N); conv1d with (K,P,L) -> (S,K,T)
        scores = F.conv1d(residual.permute(0, 2, 1), corr_weight, padding=0)  # (S,K,T)

        if pick_mode == "global":
            val_k, idx_k = scores.max(dim=1)      # (S,T)
            val_t, idx_t = val_k.max(dim=1)       # (S,)
            dZ = torch.zeros_like(Z)
            mask = (val_t > lamb)
            if mask.any():
                s_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                t_idx = idx_t[s_idx]
                k_idx = idx_k[s_idx, t_idx]
                amp = val_t[s_idx]
                dZ[s_idx, k_idx, t_idx] = amp
            else:
                break
        else:
            # NMS per (s,k) with window L
            sk = S * K
            scores_sk = scores.reshape(sk, 1, T)                 # (S*K,1,T)
            pooled = F.max_pool1d(scores_sk, kernel_size=L, stride=1)  # (S*K,1,T-L+1)
            pooled_full = F.pad(pooled, (0, L-1, 0, 0, 0, 0), value=float("-inf"))  # (S*K,1,T)
            peak_mask = (scores_sk >= pooled_full) & (scores_sk > lamb)
            peak_mask = peak_mask.view(S, K, T)
            dZ = torch.where(peak_mask, scores, torch.zeros_like(scores))  # (S,K,T)

        if (dZ.abs().sum() == 0):
            break

        # Synthesize contribution for all selected peaks
        contrib = F.conv1d(dZ, synth_weight, padding=L-1)  # (S,P,N)
        residual -= contrib.permute(0, 2, 1)               # -> (S,N,P)
        Z += dZ

        if torch.linalg.vector_norm(residual) < 1e-6:
            break

    return Z



# =========================
# Reconstruction & CDU
# =========================

def reconstruct_from_Z_phi(Z, phi, N=None):
    """
    Z:   (S, K, T) with T = N-L+1
    phi: (K, L, P)
    returns X_hat: (S, N, P)
    """
    S, K, T = Z.shape
    K2, L, P = phi.shape
    assert K2 == K
    if N is None:
        N = T + L - 1
    weight = phi.permute(2, 0, 1).flip(-1).contiguous()  # (P,K,L)
    y = F.conv1d(Z, weight, padding=L - 1)               # (S,P,N)
    return y.permute(0, 2, 1)                             # (S,N,P)


def CDU(X, Z, phi, n_iters=50, lr=1e-2):
    """
    Optimize phi with 0.5||X - sum_k z_k * phi_k||^2, s.t. ||phi_k||=1
    X:   (S,N,P)
    Z:   (S,K,T)
    phi: (K,L,P)
    returns: updated phi (K,L,P)
    """
    device = X.device
    S, N, P = X.shape
    K, L, P2 = phi.shape
    assert P2 == P

    X = to_tensor(X, device)
    Z = to_tensor(Z, device)
    phi = to_tensor(phi, device, requires_grad=True)
    phi = torch.nn.Parameter(phi)

    opt = torch.optim.Adam([phi], lr=lr)

    for _ in range(n_iters):
        opt.zero_grad()
        X_hat = reconstruct_from_Z_phi(Z, phi, N)  # (S,N,P)
        loss = 0.5 * torch.sum((X - X_hat) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            unit_norm_atoms_(phi)

    return phi.detach()

def CDU_Decorrelation(X, Z, phi, gamma=1e-2, n_iters=50, lr=1e-2):
    """
    Optimise phi avec:
    L(phi) = 0.5 * ||X - X_hat||^2 + gamma * ||phi_flat^T @ phi_flat - I||_F^2
    s.t. ||phi_k||=1

    X:   (S,N,P)
    Z:   (S,K,T)
    phi: (K,L,P)
    gamma: Regularisation de décorrélation (poids du terme d'orthogonalité)
    returns: updated phi (K,L,P)
    """
    
    # Simuler to_tensor
    def to_tensor(data, device, requires_grad=False):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        return data.requires_grad_(requires_grad)

    device = X.device
    S, N, P = X.shape
    K, L, P2 = phi.shape
    assert P2 == P

    X = to_tensor(X, device)
    Z = to_tensor(Z, device)
    phi = to_tensor(phi, device, requires_grad=True)
    phi = Parameter(phi) # Enveloppe pour l'optimiseur

    opt = torch.optim.Adam([phi], lr=lr)

    for _ in range(n_iters):
        opt.zero_grad()
        
        # 1. Terme de Reconstruction (Normalisé)
        X_hat = reconstruct_from_Z_phi(Z, phi, N)  # (S,N,P)
        loss_recon = 0.5 * torch.sum((X - X_hat) ** 2)
        
        # NORMALISATION : Diviser par le nombre total d'éléments
        # Cela transforme la perte en MSE moyenne
        num_elements = S * N * P
        loss_recon_norm = loss_recon / num_elements 
        
        # 2. Terme de Décorrélation (inchangé)
        phi_flat = phi.view(K, -1)
        gram_matrix = torch.matmul(phi_flat, phi_flat.transpose(0, 1))
        I = torch.eye(K, device=device)
        loss_decorrelation = torch.norm(gram_matrix - I, p='fro')**2
        
        # 3. Perte Totale
        # Utiliser loss_recon_norm
        loss = loss_recon_norm + gamma * loss_decorrelation
        
        # 4. Backpropagation et Mise à Jour
        loss.backward()
        opt.step()
        
        # 5. Projection (Contrainte de norme unitaire)
        with torch.no_grad():
            unit_norm_atoms_(phi) # Applique la contrainte ||phi_k||=1 in-place

    return phi.detach()

# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Synthetic data
    S, N, P = 4, 1000, 3
    K, L, M = 8, 40, 5

    # Random bounded signals
    X = torch.rand(S, N, P, device=device)

    # Init
    phi, Z, A = setInitialValues(X, K, M, L, device=device)
    print("Initialized phi:", tuple(phi.shape))  # (K,L,P)
    print("Initialized Z:", tuple(Z.shape))      # (S,K,N-L+1)

    # Batched CSC
    Z = CSC_batched(X, Z, phi, lamb=0.15, max_outer_iters=10, pick_mode="nms")
    print("Z after CSC:", tuple(Z.shape))

    # CDU
    phi = CDU(X, Z, phi, n_iters=15, lr=1e-2)
    print("phi after CDU:", tuple(phi.shape))

    # Recon error
    with torch.no_grad():
        X_hat = reconstruct_from_Z_phi(Z, phi, N)
        mse = torch.mean((X - X_hat) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")
