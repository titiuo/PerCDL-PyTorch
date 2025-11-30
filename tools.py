#####################################################################
#                                                                   #
#                           IMPORT                                  #
#                                                                   #
#####################################################################

# PyTorch de base
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# Numpy pour manipulations classiques
import numpy as np
import math

# Pour visualiser les signaux et atomes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Pour charger et manipuler des données
import pandas as pd
import json
import os
import tarfile
from urllib.request import urlretrieve

#####################################################################
#                                                                   #
#                           Helpers                                 #
#                                                                   #
#####################################################################

def orthogonalize_phi(phi):
    """
    Orthogonalise les atomes entre eux et normalise chacun.
    phi: (K, L, P)
    """
    K, L, P = phi.shape
    phi_flat = phi.reshape(K, L*P).clone()
    
    # Gram-Schmidt classique
    for i in range(K):
        for j in range(i):
            proj = (phi_flat[i] @ phi_flat[j]) / (phi_flat[j] @ phi_flat[j])
            phi_flat[i] = phi_flat[i] - proj * phi_flat[j]
        # Normalisation
        norm_i = phi_flat[i].norm()
        if norm_i > 0:
            phi_flat[i] = phi_flat[i] / norm_i
        else:
            # au cas où l'atome est nul, on le laisse tel quel
            phi_flat[i] = phi_flat[i]
    
    phi_ortho = phi_flat.reshape(K, L, P)
    return phi_ortho



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

def reconstruct_from_Z_phi_personalized(Z, phi, A, f, N=None):
    """
    Z   : (S, K, T)            with T = N - L + 1
    phi : (K, L, P)            global atoms (learned)
    A   : (S, K, M)            personalization params per subject & atom
    f   : callable (phi_k(L,P), a_sk(M,)) -> (L, P)   differentiable in phi_k
    N   : optional signal length (if None, inferred)

    returns X_hat: (S, N, P)
    """
    device = phi.device
    dtype  = phi.dtype

    S, K, T = Z.shape
    K2, L, P = phi.shape
    assert K2 == K, "K mismatch between Z and phi"

    if N is None:
        N = T + L - 1

    X_hat = torch.zeros(S, N, P, device=device, dtype=dtype)

    # Loop over subjects (weights differ per subject because A differs).
    for s in range(S):
        # Build personalized atoms for subject s: (K, L, P)
        per_atoms = []
        for k in range(K):
            per_atom = f(phi[k], A[s, k])  # (L, P)
            per_atoms.append(per_atom)
        per_atoms = torch.stack(per_atoms, dim=0)              # (K, L, P)

        # True convolution: conv1d expects (out=P, in=K, L), and conv1d is correlation,
        # so flip in time to implement convolution.
        weight = per_atoms.permute(2, 0, 1).flip(-1).contiguous()  # (P, K, L)

        # Single-subject synthesis: input (1, K, T) -> (1, P, N)
        X_hat_s = F.conv1d(Z[s:s+1], weight, padding=L-1)          # (1, P, N)
        X_hat[s] = X_hat_s.permute(0, 2, 1)[0]                     # -> (N, P)

    return X_hat


def reconstruct_from_Z_phi(Z, phi, N=None):
    """
    Reconstruction du signal X_hat = phi * Z en utilisant la Transposée de Convolution.
    
    Z:   (S, K, T) 
    phi: (K, L, P)
    returns X_hat: (S, N, P)
    """
    S, K, T = Z.shape
    K2, L, P = phi.shape
    assert K2 == K
    if N is None:
        N = T + L - 1
    
    weight_transpose = phi.permute(0, 2, 1).contiguous() 
    
    y = F.conv_transpose1d(
        Z,                           # Input: (S, K, T)
        weight_transpose,            # Weight: (K, P, L) 
        padding=0
    )
    
    if y.shape[2] > N:
         y = y[..., :N]
    
    return y.permute(0, 2, 1)                         


#####################################################################
#                                                                   #
#                           Initialization                          #
#                                                                   #
#####################################################################

def setInitialValues(X,K,M,L):
    """
    Set initial values for the parameters of PerCDL.

    Parameters:
    - K: number of common atoms in the dictionary
    - L: length of each atom (number of time samples)
    - P: number of signal dimensions (e.g., sensors or channels)
    - S: number of signals (patients or trials)

    Returns:
    - Phi: initial common dictionary (K x L x P)
    - Z: initial activations (S x K x N-L+1, initialized later)
    - A: initial personalization parameters (S x K x M)
    """
    S, N, P = X.shape  
    Phi = torch.randn(K, L, P) * 1.0
    Phi = unit_norm_atoms_(Phi)
    Z = torch.rand(S, K, N-L+1) * 0.01
    A = torch.zeros(S, K, M)

    return Phi, Z, A



#####################################################################
#                                                                   #
#                   Convolutional Sparse Coding                     #
#                                                                   #
#####################################################################


def CSC(X, Z, phi, lam, step_size=0.01, n_inner=20):
    """
    Convolutional Sparse Coding (CSC) model.

    Parameters:
    - X: input signals (S x N x P)
    - Z: activations (S x K x N-L+1)
    - phi: dictionary atoms (K x L x P)

    Returns:
    - Z: updated activations
    """

    S, N, P = X.shape  
    if 'L' in locals():
        pass
    device = X.device
    phi = phi.to(device)
    Z = Z.to(device)
    K, L, P_phi = phi.shape

    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, N-L+1), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

    phi_flip = phi.flip(dims=[1])  
    phi_conv = phi_flip.permute(0, 2, 1)  
    assert phi_conv.shape == (K, P, L), f"Phi_conv should be {(K,P,L)}, got {phi_conv.shape}"

    for s in range(S):
        x_s = X[s].permute(1, 0).unsqueeze(0)

        z_curr = Z[s].clone().detach().to(device)
        assert z_curr.shape == (K, N-L+1), f"Activation for signal {s} should be {(K, N-L+1)}, got {z_curr.shape}"
        for _ in range(n_inner):
            z_var = z_curr.clone().detach().requires_grad_(True)
            reconstructed = F.conv_transpose1d(z_var.unsqueeze(0), phi_conv, padding=0)
            loss_recon = torch.norm(x_s - reconstructed, p=2)**2
            if z_var.grad is not None:
                z_var.grad.zero_()
            loss_recon.backward()
            grad_z = z_var.grad

            with torch.no_grad():
                grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e6, neginf=-1e6)
                grad_z = grad_z.clamp(min=-1e3, max=1e3)

                u = z_curr - step_size * grad_z

                z_next = torch.clamp(u - lam * step_size, min=0.0)

                z_next = torch.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=0.0)

                z_curr = z_next

        Z[s] = z_curr.detach()

    return Z

def CSC_l0(X, Z, phi, lam, step_size=0.01, n_inner=20):
    """
    Convolutional Sparse Coding (CSC) model using Iterative Hard Thresholding (IHT) 
    for L0 regularization and non-negativity constraint.

    Parameters:
    - X: input signals (S x N x P)
    - Z: activations (S x K x N-L+1)
    - phi: dictionary atoms (K x L x P)
    - lam: L0 regularization parameter (lambda)
    - step_size: Gradient step size (tau)
    - n_inner: Number of inner proximal gradient iterations
    
    Returns:
    - Z: updated activations
    """

    S, N, P = X.shape  
    device = X.device
    
    phi = phi.to(device)
    Z = Z.to(device)
    K, L, P_phi = phi.shape

    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, N-L+1), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

    phi_flip = phi.flip(dims=[1]) 
    phi_conv = phi_flip.permute(0, 2, 1)  

    T = lam * step_size 

    for s in range(S):
        x_s = X[s].permute(1, 0).unsqueeze(0) 

        z_curr = Z[s].clone().detach().to(device) 
        
        for _ in range(n_inner):

            z_var = z_curr.clone().detach().requires_grad_(True)
            
            reconstructed = F.conv_transpose1d(z_var.unsqueeze(0), phi_conv, padding=0)
            
            loss_recon = torch.norm(x_s - reconstructed, p=2)**2
            
            if z_var.grad is not None:
                z_var.grad.zero_()
            loss_recon.backward()
            grad_z = z_var.grad

            with torch.no_grad():
                grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e6, neginf=-1e6)

                grad_z = grad_z.squeeze(0) 

                u = z_curr - step_size * grad_z
                z_next = torch.where(u >= T, u, torch.zeros_like(u))

                z_next = torch.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=0.0)

                z_curr = z_next

        Z[s] = z_curr.detach()

    return Z

def CSC_l0_NMS(X, Z, phi, lam, step_size=0.01, n_inner=20, nms_radius=2):
    """
    Convolutional Sparse Coding (CSC) model using Iterative Hard Thresholding (IHT) 
    for L0 regularization, non-negativity, ET COMPACITÉ SPATIALE (NMS).

    Parameters:
    - X: input signals (S x N x P)
    - Z: activations (S x K x N-L+1)
    - phi: dictionary atoms (K x L x P)
    - lam: L0 regularization parameter (lambda)
    - step_size: Gradient step size (tau)
    - n_inner: Number of inner proximal gradient iterations
    - nms_radius: Rayon de voisinage pour la Non-Maximal Suppression (e.g., 1 position à gauche/droite)
    
    Returns:
    - Z: updated activations
    """

    S, N, P = X.shape  
    device = X.device
    
    phi = phi.to(device)
    Z = Z.to(device)
    K, L, P_phi = phi.shape
    T_Z = Z.shape[2] 

    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, T_Z), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

   
    phi_flip = phi.flip(dims=[1]) 
    phi_conv = phi_flip.permute(0, 2, 1) 

    T = lam * step_size 

    for s in range(S):
        x_s = X[s].permute(1, 0).unsqueeze(0) 

        z_curr = Z[s].clone().detach().to(device) 
        
        for _ in range(n_inner):
            
            z_var = z_curr.clone().detach().requires_grad_(True)
            
            reconstructed = F.conv_transpose1d(z_var.unsqueeze(0), phi_conv, padding=0)
            
            loss_recon = torch.norm(x_s - reconstructed, p=2)**2
            
            if z_var.grad is not None:
                z_var.grad.zero_()
            loss_recon.backward()
            grad_z = z_var.grad

            with torch.no_grad():
                grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e6, neginf=-1e6)
                grad_z = grad_z.squeeze(0) 

                u = z_curr - step_size * grad_z

                u_seuil = torch.where(u >= T, u, torch.zeros_like(u))
                
                z_next = torch.zeros_like(u_seuil)
                
                is_peak = torch.ones_like(u_seuil, dtype=torch.bool)
                
                for r in range(1, nms_radius + 1):
                    u_left_shifted = F.pad(u_seuil[:, :-r], (r, 0), 'constant', 0)
                    is_peak &= (u_seuil > u_left_shifted)

                    u_right_shifted = F.pad(u_seuil[:, r:], (0, r), 'constant', 0)
                    is_peak &= (u_seuil >= u_right_shifted)

                is_peak &= (u_seuil > 0)
                
                z_next[is_peak] = u_seuil[is_peak]
                
                z_next = torch.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=0.0)

                z_curr = z_next

        Z[s] = z_curr.detach()

    return Z


import torch

def CSC_L0_DP(X, Z, phi, lam, eps_reg=1e-6):
    """
    Corrected CSC_L0 DP implementation following the paper's formulas.
    X: (S, N, P)
    Z: (S, K, T_Z) used only for shape
    phi: (K, L, P)
    lam: scalar (must be on same scale as ||P y||^2)
    """
    phi = orthogonalize_phi(phi)
    S, N, P = X.shape
    device = X.device
    K, L, P_phi = phi.shape
    assert P == P_phi
    T_Z = N - L + 1

    D_matrix = phi.reshape(K, L * P).T.to(device)  # (L*P) x K

    G = D_matrix.T @ D_matrix  # K x K
    reg = eps_reg * torch.eye(K, device=device)
    try:
        G_inv = torch.linalg.inv(G + reg)
    except Exception:
        G_inv = torch.linalg.pinv(G + reg)

    P_tilde = G_inv @ D_matrix.T  # K x (L*P)

    Z_out = torch.zeros(S, K, T_Z, device=device)

    for s in range(S):
        x_s = X[s]  # (N, P)
        optimal_coeffs = torch.zeros(T_Z, K, device=device)
        reconstruction_norm_sq = torch.zeros(T_Z, device=device)

        for t in range(T_Z):
            y_block = x_s[t:t+L, :].reshape(-1)  # (L*P,)
            u_t = P_tilde @ y_block
            optimal_coeffs[t] = u_t

            reconstruction = D_matrix @ u_t
            reconstruction_norm_sq[t] = torch.sum(reconstruction ** 2)

        segment_costs = lam - reconstruction_norm_sq  # length T_Z

        V = torch.zeros(T_Z + 1, device=device)
        t_prev_opt = torch.zeros(T_Z + 1, dtype=torch.long, device=device)

        for t in range(L, T_Z + 1):
            idx_Z = t - L
            cost_no_activation = V[t - 1]
            cost_with_activation = V[idx_Z] + segment_costs[idx_Z]
            if cost_with_activation < cost_no_activation:
                V[t] = cost_with_activation
                t_prev_opt[t] = idx_Z  
            else:
                V[t] = cost_no_activation
                t_prev_opt[t] = t_prev_opt[t - 1]

        current_t = T_Z
        while current_t >= L:
            if t_prev_opt[current_t] == current_t - L:
                t_start = current_t - L
                Z_out[s, :, t_start] = optimal_coeffs[t_start].clamp(min=0.0)  # clamp if you want non-negativity
                current_t = t_start
            else:
                current_t -= 1

    return Z_out.detach()


def CSC_L0_DP_AMO(X, Z, phi, lam):
    """
    Convolutional Sparse Coding (CSC) with AtMostOneActivation constraint.
    (CSC-L0 regime 2 from Truong & Moreau, 2024)

    Parameters:
    - X: input signals (S x N x P)
    - Z: activations (S x K x N-L+1) (Used for shape/dimension initialization only)
    - phi: dictionary atoms (K x L x P)
    - lam: L0 regularization parameter (lambda), threshold on squared correlation.

    Returns:
    - Z: The optimal activations (S x K x N-L+1), non-zero in at most one entry per column.
    """

    S, N, P = X.shape
    device = X.device
    K, L, P_phi = phi.shape
    T_Z = N - L + 1
    assert P == P_phi

    D_matrix = phi.reshape(K, L * P).to(device)
    
    Z_out = torch.zeros(S, K, T_Z, device=device)

    for s in range(S):
        x_s = X[s].T.reshape(-1) # (P*N)
        
        correlations_sq = torch.zeros(T_Z, K, device=device) 
        optimal_coeffs = torch.zeros(T_Z, K, device=device) # T_Z x K
        
        for t in range(T_Z):
            y_block = x_s[t*P : (t+L)*P] 
            
            corr_k = torch.matmul(D_matrix, y_block) 
            
            optimal_coeffs[t] = corr_k.clamp(min=0.0) 
            correlations_sq[t] = corr_k**2

        max_corr_sq, best_k_idx = correlations_sq.max(dim=1)
        
        segment_costs = lam - max_corr_sq
        
        V = torch.zeros(T_Z + 1, device=device)
        t_prev_opt = torch.zeros(T_Z + 1, dtype=torch.long, device=device) 
        V[:L] = 0.0

        for t in range(L, T_Z + 1):
            idx_Z = t - L 
            
            cost_no_activation = V[t-1]
            
            cost_with_activation = V[idx_Z] + segment_costs[idx_Z]
            
            if cost_with_activation < cost_no_activation:
                V[t] = cost_with_activation
                t_prev_opt[t] = t - L 
            else:
                V[t] = cost_no_activation
                t_prev_opt[t] = t_prev_opt[t-1] 

        
        current_t = T_Z
        while current_t >= L:
            if t_prev_opt[current_t] == current_t - L:
                t_start = current_t - L
                idx_Z = t_start
                
                best_k = best_k_idx[idx_Z]
                
                Z_out[s, best_k, idx_Z] = optimal_coeffs[idx_Z, best_k]
                
                current_t = t_start
            else:
                current_t -= 1
    
    return Z_out.detach()



#####################################################################
#                                                                   #
#                  Convolutional dictionary update                  #
#                                                                   #
#####################################################################

def CDU(X, Z, phi, n_iters=500, lr=5e-1):
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


#####################################################################
#                                                                   #
#                    Transformation Function                        #
#                                                                   #
#####################################################################

def time_warping_f(phi_k, a_k_s, sigma=0.1):
    """
    Time-warping 1D strict (interpolation linéaire)
    Pas de grid_sample, pas de lissage transversal.
    
    phi_k: (L,P)
    a_k_s: (M,)
    return: (L,P)
    """
    L, P = phi_k.shape
    M = a_k_s.shape[0]
    device = phi_k.device

    # timeline
    t_i = torch.linspace(0, 1, L, device=device)

    # Fourier-based displacement (comme ta version originale)
    w = torch.arange(1, M + 1, device=device, dtype=torch.float32)
    b_w = torch.sin(w[None, :] * math.pi * t_i[:, None]) / (w[None, :] * math.pi)
    displacement = b_w @ a_k_s

    # warp map
    psi_a_t = torch.clamp(t_i + displacement, 0.0, 1.0)
    x = psi_a_t * (L - 1)

    # indices d'interpolation linéaire
    x0 = torch.floor(x).long()
    x1 = torch.clamp(x0 + 1, max=L - 1)
    alpha = (x - x0.float()).unsqueeze(-1)   # (L,1)

    # lookup propre, pas de broadcast bug
    phi0 = phi_k[x0]          # (L,P)
    phi1 = phi_k[x1]          # (L,P)

    warped_phi = phi0 * (1 - alpha) + phi1 * alpha
    return warped_phi



#####################################################################
#                                                                   #
#                    Individual parameters update                   #
#                                                                   #
#####################################################################

def IPU(X, Z, Phi, A, f=time_warping_f,n_iters=50, lr=1e-2, sigma=0.01):
    """
    IPU : optimize A (time-warp parameters) with
    0.5 * ||X - sum_k z_k * f(phi_k, a_k^s)||^2

    Inputs:
        X:    (S, N, P)
        Z:    (S, K, T)
        Phi:  (K, L, P)
        A:    (S, K, M)
        f:    function (phi_k, a_k_s) -> (L, P)
        n_iters: number of gradient steps
        lr:   learning rate
        sigma: smoothing param for f
    Output:
        A_new: updated A (S, K, M)
    """
    device = X.device
    S, N, P = X.shape
    S2, K, M = A.shape
    K2, L, P2 = Phi.shape
    assert S2 == S and K2 == K and P2 == P, "Shape mismatch between X, Phi, A"

    X = X.to(device)
    Z = Z.to(device)
    Phi = Phi.to(device)
    A = torch.nn.Parameter(A.clone().detach().to(device))

    opt = torch.optim.Adam([A], lr=lr)

    for _ in range(n_iters):
        opt.zero_grad()

        warped_Phi = torch.zeros((S, K, L, P), device=device)
        for s in range(S):
            for k in range(K):
                warped_Phi[s, k] = f(Phi[k], A[s, k], sigma=sigma)

        X_hat = torch.zeros((S, N, P), device=device)
        for s in range(S):
            weight = warped_Phi[s].permute(2, 0, 1).flip(-1).contiguous()  # (P,K,L)
            X_hat[s] = F.conv1d(Z[s].unsqueeze(0), weight, padding=L - 1).permute(0, 2, 1)[0, :N, :]

        loss = 0.5 * torch.sum((X - X_hat) ** 2)
        loss.backward()
        opt.step()

    return A.detach()

#####################################################################
#                                                                   #
#                          Personalized CDU                         #
#                                                                   #
#####################################################################
    
def inverse_warp_patch(y_patch, a_k_s, L=None):
    """
    Inverse-warp a single patch y_patch of shape (L, P) back to canonical frame.
    Uses the same displacement basis as time_warping_f and numpy.interp to invert psi.
    Returns aligned_patch (L, P) on same dtype/device as input.
    """
    # y_patch : (L, P) tensor
    device = y_patch.device
    dtype = y_patch.dtype
    if L is None:
        L = y_patch.shape[0]
    M = a_k_s.shape[0]

    # timeline [0,1]
    t_i = torch.linspace(0.0, 1.0, L, device=device, dtype=dtype)

    # build displacement like in time_warping_f
    w = torch.arange(1, M + 1, device=device, dtype=dtype)
    # b_w: (L, M)
    b_w = torch.sin(w[None, :] * math.pi * t_i[:, None]) / (w[None, :] * math.pi)
    displacement = b_w @ a_k_s  # (L,)

    psi = torch.clamp(t_i + displacement, 0.0, 1.0)  # (L,)

    # we'll invert psi: for canonical grid u_j = t_i we want t = psi^{-1}(u)
    psi_cpu = psi.detach().cpu().numpy()
    t_cpu = t_i.detach().cpu().numpy()
    u_cpu = t_cpu  # same grid

    # if psi not strictly monotonic, np.interp will still do something (piecewise)
    t_of_u = np.interp(u_cpu, psi_cpu, t_cpu)  # gives t corresponding to each u

    # map to sample locations in [0, L-1]
    x = t_of_u * (L - 1)  # float positions
    # linear interpolation of y_patch along axis 0
    y_np = y_patch.detach().cpu().numpy()  # (L, P)
    L_int, P = y_np.shape

    # sample for each dim p
    aligned = np.empty((L, P), dtype=y_np.dtype)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, L_int - 1)
    alpha = (x - x0).reshape(-1, 1)

    aligned = (1.0 - alpha) * y_np[x0, :] + alpha * y_np[x1, :]

    aligned_t = torch.from_numpy(aligned).to(device=device, dtype=dtype)
    return aligned_t


def PerCDU(X, Z, phi, A, alpha_blend=0.02, lambda_reg=5e-3):
    """
    PerCDU stable sans min_count_threshold.
    
    X : (S,N,P)
    Z : (S,K,T)
    phi : (K,L,P)
    A : (S,K,M)
    func : time-warping function f(phi_k, a_sk)
    alpha_blend : blending step to control atom drift
    lambda_reg : ridge regularization toward previous phi
    
    Retourne :
        final_phi : (K,L,P)  (detach, normalized)
    """
    device = X.device
    dtype  = X.dtype

    S, N, P = X.shape
    K, L, P2 = phi.shape
    assert P2 == P

    X_t  = X.to(device)
    Z_t  = Z.to(device)
    A_t  = A.to(device)
    phi_old = phi.to(device)

    new_phi_est = torch.zeros_like(phi_old, device=device, dtype=dtype)

    T = Z_t.shape[2]  # T = N - L + 1

    # ---- Compute new phi (closed-form average of inverse-warped patches) ----
    for k in range(K):
        numer = torch.zeros((L, P), device=device, dtype=dtype)
        denom = 0.0

        for s in range(S):
            z_sk = Z_t[s, k]  # (T,)
            idxs = torch.nonzero(z_sk > 0, as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue

            a_sk = A_t[s, k]  # (M,)

            for t_idx in idxs.tolist():
                start = t_idx
                end = t_idx + L
                if end > N:
                    continue
                y_patch = X_t[s, start:end, :]    # (L, P)

                aligned = inverse_warp_patch(y_patch, a_sk, L=L)  # (L,P)

                weight = float(z_sk[t_idx].detach().cpu().item())
                numer += weight * aligned
                denom += weight

        # ---- Regularization towards old phi ----
        if denom > 0:
            numer = numer + lambda_reg * phi_old[k]
            denom = denom + lambda_reg

            new_phi_est[k] = numer / (denom + 1e-12)
        else:
            # no occurrences: keep the previous atom
            new_phi_est[k] = phi_old[k].clone()

    # ---- Normalize new_phi_est ----
    norms = new_phi_est.view(K, -1).norm(p=2, dim=1).view(K, 1, 1)
    norms = torch.clamp(norms, min=1e-12)
    new_phi_est = new_phi_est / norms

    # ---- Blend with previous atoms (stabilization) ----
    final_phi = (1.0 - alpha_blend) * phi_old + alpha_blend * new_phi_est

    # ---- Normalize final phi ----
    norms = final_phi.view(K, -1).norm(p=2, dim=1).view(K, 1, 1)
    norms = torch.clamp(norms, min=1e-12)
    final_phi = final_phi / norms

    return final_phi.detach()


#####################################################################
#                                                                   #
#                          Personalized CDL                         #
#                                                                   #
#####################################################################


def PerCDL(X,nb_atoms,D=3,W=10,atoms_length=50,func=time_warping_f,lambda_=0.01,n_iters=100,n_perso=100):
    
    K = nb_atoms
    L = atoms_length
    M=D*W

    Phi, Z, A = setInitialValues(X,K,M,L)
 

    for it in range(n_iters):
        Z = CSC_L0_DP(X, Z, Phi, lambda_)
        Phi = CDU(X, Z, Phi,lr=5e-2)


    for it in range(n_perso):
        A = IPU(X, Z, Phi, A,f=func)
        Z = CSC_L0_DP(X, Z, Phi, lambda_)
        Phi = PerCDU(X, Z, Phi, A,func=func)

    return A,Z,Phi

def Personalization(X,A,Z,Phi,lambda_=0.01,func=time_warping_f,n_perso=100):

    for it in range(n_perso):
        A = IPU(X, Z, Phi, A,f=func)
        Z = CSC_L0_DP(X, Z, Phi, lambda_)
        Phi = PerCDU(X, Z, Phi, A)

    return A,Z,Phi

#####################################################################
#                                                                   #
#                                CDL                                #
#                                                                   #
#####################################################################

def CDL(X,nb_atoms,D=3,W=10,atoms_length=50,lambda_=0.01,n_iters=100):
    
    K = nb_atoms
    L = atoms_length
    M=D*W

    Phi, Z, A = setInitialValues(X,K,M,L)
 

    for it in range(n_iters):
        Z = CSC_L0_DP(X, Z, Phi, lambda_)
        Phi = CDU(X, Z, Phi,lr=5e-2)

    return Z,Phi,A