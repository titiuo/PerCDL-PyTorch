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

def to_tensor(x, device, requires_grad: bool = False, dtype=torch.float32):
    """Safe wrapping to tensor without copy-construct warnings."""
    if torch.is_tensor(x):
        t = x.detach().clone().to(device=device, dtype=dtype)
        t.requires_grad_(requires_grad)
        return t
    t = torch.as_tensor(x, dtype=dtype, device=device)
    t.requires_grad_(requires_grad)
    return t


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


def unit_norm_atoms_(phi, eps=1e-12):
    """Project each atom (K,L,P) to unit l2 norm."""
    K = phi.shape[0]
    flat = phi.view(K, -1)
    norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min_(eps)
    phi.copy_((flat / norms).view_as(phi))
    return phi

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
    S, N, P = X.shape  # X is S x N x P
    # Initialisation des atomes (petit bruit aléatoire plutôt que zéros)
    # utiliser une petite variance pour éviter des atomes exactement nuls
    Phi = torch.randn(K, L, P) * 1e-2
    # normaliser chaque atome pour respecter la contrainte ||phi_k||_2 = 1
    Phi = unit_norm_atoms_(Phi)
    # Initialisation des coefficients
    Z = torch.rand(S, K, N-L+1) * 0.01  
    # Initialisation des poids
    A = torch.zeros(S, K, M)

    return Phi, Z, A

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

    S, N, P = X.shape  # X is S x N x P
    # sécurité : s'assurer que l'atome n'est pas plus long que le signal
    if 'L' in locals():
        pass
    # s'assurer que les tenseurs sont sur le même device
    device = X.device
    phi = phi.to(device)
    Z = Z.to(device)
    K, L, P_phi = phi.shape

    # Vérification des dimensions
    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, N-L+1), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

    # Precompute flipped+permuted phi for conv_transpose
    phi_flip = phi.flip(dims=[1])  # convolution flip
    phi_conv = phi_flip.permute(0, 2, 1)  # K x P x L
    assert phi_conv.shape == (K, P, L), f"Phi_conv should be {(K,P,L)}, got {phi_conv.shape}"

    for s in range(S):
        # Permute to channels first for conv1d: 1 x P x N
        x_s = X[s].permute(1, 0).unsqueeze(0)

        # work with a local copy of z (no grad) and perform proximal gradient iterations
        z_curr = Z[s].clone().detach().to(device)
        assert z_curr.shape == (K, N-L+1), f"Activation for signal {s} should be {(K, N-L+1)}, got {z_curr.shape}"
        for _ in range(n_inner):
            # create a differentiable version to compute gradient of reconstruction term only
            z_var = z_curr.clone().detach().requires_grad_(True)
            reconstructed = F.conv_transpose1d(z_var.unsqueeze(0), phi_conv, padding=0)
            # reconstruction loss (without l1)
            loss_recon = torch.norm(x_s - reconstructed, p=2)**2
            # compute gradient wrt z_var
            if z_var.grad is not None:
                z_var.grad.zero_()
            loss_recon.backward()
            grad_z = z_var.grad

            with torch.no_grad():
                # safe-guard gradient values
                grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e6, neginf=-1e6)
                grad_z = grad_z.clamp(min=-1e3, max=1e3)

                # gradient step (on reconstruction term)
                u = z_curr - step_size * grad_z

                # proximal operator for l1 with non-negativity: soft-threshold followed by clamp
                # For non-negative activations, prox simplifies to: z = max(0, u - lam*step)
                z_next = torch.clamp(u - lam * step_size, min=0.0)

                # ensure numerical stability
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

    S, N, P = X.shape  # X is S x N x P
    device = X.device
    
    # Assurer que les tenseurs sont sur le même device
    phi = phi.to(device)
    Z = Z.to(device)
    K, L, P_phi = phi.shape

    # Vérifications des dimensions
    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, N-L+1), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

    # Précalculer phi 'flippé' et permuté pour F.conv_transpose1d
    # Flipper sur la dimension du temps/longueur (dims=[1])
    phi_flip = phi.flip(dims=[1]) 
    # Permuter pour le format attendu par conv_transpose1d: (K, P, L) -> (out_channels, in_channels, kernel_size)
    phi_conv = phi_flip.permute(0, 2, 1)  # K x P x L

    # Calcul du seuil T pour le Hard-Thresholding
    # T = lam * step_size est une simplification courante pour IHT.
    # On pourrait aussi utiliser T = sqrt(2 * lam * step_size)
    T = lam * step_size 

    for s in range(S):
        # Permuter pour le format Pytorch Conv1d: (N x P) -> (P x N)
        x_s = X[s].permute(1, 0).unsqueeze(0) # 1 x P x N

        # Copie locale de Z pour les itérations de PGM (pas de suivi de gradient direct)
        z_curr = Z[s].clone().detach().to(device) # K x N-L+1
        
        for _ in range(n_inner):
            # 1. Calcul du Gradient du terme de Reconstruction
            
            # Créer une variable pour le calcul du gradient
            z_var = z_curr.clone().detach().requires_grad_(True)
            
            # Reconstruire le signal: (1 x K x N-L+1) * (K x P x L) -> (1 x P x N)
            reconstructed = F.conv_transpose1d(z_var.unsqueeze(0), phi_conv, padding=0)
            
            # Perte de reconstruction (L2^2)
            loss_recon = torch.norm(x_s - reconstructed, p=2)**2
            
            # Calculer le gradient wrt z_var
            if z_var.grad is not None:
                z_var.grad.zero_()
            loss_recon.backward()
            grad_z = z_var.grad

            with torch.no_grad():
                # Safe-guard gradient values
                grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e6, neginf=-1e6)

                # Le gradient est (1 x K x N-L+1). On le ramène à (K x N-L+1) pour u.
                grad_z = grad_z.squeeze(0) 

                # 2. Étape du Gradient : u = z_curr - step_size * grad_z
                u = z_curr - step_size * grad_z

                # 3. Opérateur Proximal pour L0 (Hard-Thresholding) avec non-négativité
                # On met à zéro tout élément de u qui est inférieur à T.
                # L'opérateur garantit aussi z_next >= 0 puisque u est comparé à T (T > 0 si lam > 0)
                z_next = torch.where(u >= T, u, torch.zeros_like(u))

                # Assurer la stabilité numérique
                z_next = torch.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=0.0)

                z_curr = z_next

        # Mettre à jour Z avec le résultat de l'itération PGM
        Z[s] = z_curr.detach()

    return Z

def CSC_l0_NMS(X, Z, phi, lam, step_size=0.01, n_inner=20, nms_radius=1):
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

    S, N, P = X.shape  # X is S x N x P
    device = X.device
    
    # Assurer que les tenseurs sont sur le même device
    phi = phi.to(device)
    Z = Z.to(device)
    K, L, P_phi = phi.shape
    T_Z = Z.shape[2] # Taille temporelle de Z

    # Vérifications des dimensions (laissées pour la robustesse)
    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, T_Z), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

    # Précalculer phi 'flippé' et permuté pour F.conv_transpose1d
    phi_flip = phi.flip(dims=[1]) 
    phi_conv = phi_flip.permute(0, 2, 1)  # K x P x L

    # Calcul du seuil T pour le Hard-Thresholding
    T = lam * step_size 

    for s in range(S):
        # Permuter pour le format Pytorch Conv1d: (N x P) -> (P x N)
        x_s = X[s].permute(1, 0).unsqueeze(0) # 1 x P x N

        # Copie locale de Z pour les itérations de PGM
        z_curr = Z[s].clone().detach().to(device) # K x T_Z
        
        for _ in range(n_inner):
            # 1. Calcul du Gradient du terme de Reconstruction
            
            z_var = z_curr.clone().detach().requires_grad_(True)
            
            # Reconstruire le signal
            reconstructed = F.conv_transpose1d(z_var.unsqueeze(0), phi_conv, padding=0)
            
            # Perte de reconstruction (L2^2)
            loss_recon = torch.norm(x_s - reconstructed, p=2)**2
            
            # Calculer le gradient wrt z_var
            if z_var.grad is not None:
                z_var.grad.zero_()
            loss_recon.backward()
            grad_z = z_var.grad

            with torch.no_grad():
                # Safe-guard gradient values
                grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e6, neginf=-1e6)
                grad_z = grad_z.squeeze(0) # Ramener à (K x T_Z)

                # 2. Étape du Gradient : u = z_curr - step_size * grad_z
                u = z_curr - step_size * grad_z

                # 3. Opérateur Proximal (Hard-Thresholding + Non-Negativity + NMS)
                
                # 3a. Hard-Thresholding (Seuillage) et Non-Négativité
                # Seuls les éléments > T sont conservés.
                u_seuil = torch.where(u >= T, u, torch.zeros_like(u))
                
                # 3b. Non-Maximal Suppression (NMS) pour forcer les diracs isolés
                
                # Initialiser z_next à zéro
                z_next = torch.zeros_like(u_seuil)
                
                # Créer des masques pour les comparaisons gauche/droite
                is_peak = torch.ones_like(u_seuil, dtype=torch.bool)
                
                # Comparaison avec les voisins pour un rayon de 'nms_radius'
                for r in range(1, nms_radius + 1):
                    # Comparaison avec les voisins de GAUCHE (décalage vers la droite)
                    # Ex: si r=1, u[t] doit être > u[t-1]
                    u_left_shifted = F.pad(u_seuil[:, :-r], (r, 0), 'constant', 0)
                    is_peak &= (u_seuil > u_left_shifted)

                    # Comparaison avec les voisins de DROITE (décalage vers la gauche)
                    # Ex: si r=1, u[t] doit être >= u[t+1]
                    u_right_shifted = F.pad(u_seuil[:, r:], (0, r), 'constant', 0)
                    is_peak &= (u_seuil >= u_right_shifted)

                # S'assurer que les valeurs non-nulles sont considérées (évite les faux pics à 0)
                is_peak &= (u_seuil > 0)
                
                # Appliquer le masque de pic à la version seuillée de u
                z_next[is_peak] = u_seuil[is_peak]
                
                # Assurer la stabilité numérique
                z_next = torch.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=0.0)

                z_curr = z_next

        # Mettre à jour Z avec le résultat de l'itération PGM
        Z[s] = z_curr.detach()

    return Z

def CDU(X,Z,phi,step_size=0.01):
    """
    Convolutional Dictionary Update (CDU) model.

    Parameters:
    - X: input signals (S x P x N)
    - Z: activations (S x K x N-L+1)
    - phi: dictionary atoms (K x L x P)

    Returns:
    - phi: updated dictionary atoms
    """

    S, N, P = X.shape
    K, L, P_phi = phi.shape

    # s'assurer que les tenseurs sont sur le même device
    device = X.device
    phi = phi.to(device)
    Z = Z.to(device)

    # Vérification des dimensions
    assert P == P_phi, f"Mismatch in signal channels: X has {P}, phi has {P_phi}"
    assert Z.shape == (S, K, N-L+1), f"Z should have shape (S,K,N-L+1), got {Z.shape}"

    # utiliser un tenseur pour accumuler la loss sur le bon device
    loss = torch.tensor(0.0, device=device)

    # Travailler sur une copie qui permet le calcul du gradient
    phi_var = phi.clone().detach().requires_grad_(True)

    for s in range(S):
        x_s = X[s].permute(1, 0).unsqueeze(0)  # 1 x P x N
        z_s = Z[s].unsqueeze(0)               # 1 x K x N-L+1
        phi_conv = phi_var.permute(0, 2, 1)   # K x P x L
        reconstructed = F.conv_transpose1d(z_s, phi_conv, padding=0)
        loss = loss + torch.norm(x_s - reconstructed, p=2)**2

    # Backpropagation
    loss.backward()

    with torch.no_grad():
        if phi_var.grad is not None:
            g = torch.nan_to_num(phi_var.grad, nan=0.0, posinf=1e6, neginf=-1e6)
            g = g.clamp(min=-1e3, max=1e3)
            phi_var = phi_var - step_size * g

        # Normalisation de chaque atome avec epsilon pour éviter division par zéro
        phi_var = unit_norm_atoms_(phi_var)

        # retourner la version detachée sur le device d'origine
        phi_out = phi_var.detach().to(device)

        # nettoyer gradients si nécessaire
        if phi_var.grad is not None:
            phi_var.grad.zero_()

    return phi_out


def time_warping_f(phi_k, a_k_s, sigma=0.01):
    """
    Applique la transformation de time warping f sur un atome phi_k avec un paramètre a_k_s.

    Inputs:
        phi_k: Tensor de taille (L, P)
        a_k_s: Tensor de taille (M,)
        sigma: float, paramètre de lissage pour l'interpolation

    Output:
        warped_phi: Tensor de taille (L, P)
    """
    L, P = phi_k.shape
    M = a_k_s.shape[0]
    
    assert phi_k.ndim == 2, f"phi_k doit être de dimension 2, got {phi_k.ndim}"
    assert a_k_s.ndim == 1, f"a_k_s doit être de dimension 1, got {a_k_s.ndim}"

    device = phi_k.device

    # 1. Temps échantillonnés uniformes
    t_i = torch.linspace(0, 1, L, device=device)  # shape (L,)

    # 2. Construction du time warping ψ_a(t)
    w = torch.arange(1, M + 1, device=device, dtype=torch.float32)
    b_w = torch.sin(w[None, :] * math.pi * t_i[:, None]) / (w[None, :] * math.pi)
    displacement = b_w @ a_k_s
    psi_a_t = t_i + displacement
    psi_a_t = torch.clamp(psi_a_t, 0.0, 1.0)

    # 3. Interpolation linéaire différentiable
    # Échelle psi_a_t dans [0, L-1]
    x = psi_a_t * (L - 1)
    x0 = torch.floor(x).long()
    x1 = torch.clamp(x0 + 1, max=L - 1)
    alpha = (x - x0.float()).unsqueeze(-1)  # poids pour l'interpolation linéaire

    # Récupérer les valeurs correspondantes
    phi0 = phi_k[x0]      # (L, P)
    phi1 = phi_k[x1]      # (L, P)

    warped_phi = phi0 * (1 - alpha) + phi1 * alpha  # interpolation linéaire

    return warped_phi

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

        # On reconstruit tous les atomes déformés f(phi_k, a_k^s)
        warped_Phi = torch.zeros((S, K, L, P), device=device)
        for s in range(S):
            for k in range(K):
                warped_Phi[s, k] = f(Phi[k], A[s, k], sigma=sigma)

        # Reconstruction : X_hat[s] = sum_k z_k^s * warped_phi_k^s
        # Pour utiliser la même fonction que CDU, on doit moyenner sur s
        X_hat = torch.zeros((S, N, P), device=device)
        for s in range(S):
            # Convolution 1D sur le signal s, en sommant sur K
            weight = warped_Phi[s].permute(2, 0, 1).flip(-1).contiguous()  # (P,K,L)
            X_hat[s] = F.conv1d(Z[s].unsqueeze(0), weight, padding=L - 1).permute(0, 2, 1)[0, :N, :]

        # Calcul de la perte
        loss = 0.5 * torch.sum((X - X_hat) ** 2)
        loss.backward()
        opt.step()

    return A.detach()

def phi_hat(phi, A, func=time_warping_f, sigma=0.01):
    """
    Applique la transformation de time warping f sur chaque atome phi_k avec le paramètre a_k.

    Inputs:
        phi: Tensor de taille (K, L, P)   -- dictionnaire commun
        a:   Tensor de taille (K, M)      -- paramètres de time warping pour chaque atome
        func: function (phi_k, a_k) -> warped_phi_k
        sigma: float                     -- paramètre de lissage pour l'interpolation

    Output:
        warped_Phi: Tensor de taille (K, L, P)
    """
    K, L, P = phi.shape
    M = A.shape[1]
    
    assert phi.ndim == 3, f"phi doit être de dimension 3, got {phi.ndim}"
    assert A.ndim == 3, f"A doit être de dimension 3, got {A.ndim}"

    warped_Phi = torch.zeros_like(phi)  # shape (K, L, P)
    
    for k in range(K):
        for s in range(A.shape[0]):
            warped_Phi[k,k,:] = func(phi[k], A[s,k,:], sigma=sigma)

    return warped_Phi
    
def PerCDU(X,Z,phi,A,func=time_warping_f,sigma=0.01,n_iters=50, lr=1e-2):
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
        X_hat = reconstruct_from_Z_phi_personalized(Z, phi, A, f=func, N=N) # (S,N,P)
        loss = 0.5 * torch.sum((X - X_hat) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            unit_norm_atoms_(phi)

    return phi.detach()

if __name__ == "__main__":
    X = torch.randn(5, 100, 3)  # Example input
    K = 2
    L = 20
    M=10
    lambda_ = 0.1
    Phi,Z,A = setInitialValues(X,K,M,L)
    Z_updated = CSC(X, Z, Phi, lambda_)
    print("Updated Z shape:", Z_updated.shape)
    Phi_updated = CDU(X, Z_updated, Phi)
    print("Updated Phi shape:", Phi_updated.shape)
    A_updated = IPU(X, Z_updated, Phi_updated, A)
    print("Updated A shape:", A_updated.shape)
    Z = CSC(X, Z, Phi_updated, lambda_)
    Phi = PerCDU(X, Z, Phi_updated, A)
    print("PerCDU Phi shape:", Phi.shape)