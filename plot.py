import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tools import time_warping_f

def check_tensors(X, Z, Phi):
    """Vérifie l'existence des tenseurs et les copie sur CPU."""
    if X is None or Z is None or Phi is None:
        raise ValueError("Les tenseurs Phi, Z, et X ne doivent pas être None.")

    # S'assurer que les tenseurs sont en place
    # Cette vérification est moins nécessaire si on passe les tenseurs en arguments
    # mais reste une bonne pratique pour les vérifier.
    
    # Copier sur CPU pour la visualisation
    Phi_cpu = Phi.detach().cpu()
    Z_cpu = Z.detach().cpu()
    X_cpu = X.detach().cpu()

    # quick NaN check
    if torch.isnan(Phi_cpu).any() or torch.isnan(Z_cpu).any() or torch.isnan(X_cpu).any():
        print("Attention : NaN détecté dans Phi / Z / X !")
        
    return X_cpu, Z_cpu, Phi_cpu

def plot_atoms(Phi: torch.Tensor):
    """
    1) Affiche les atomes (Phi) par canal.
    
    Args:
        Phi (torch.Tensor): Tenseur des atomes de forme (K, L, P).
    """
    Phi_cpu = Phi.detach().cpu()
    K, L, P = Phi_cpu.shape
    
    print("--- 1) Affichage des atomes (Phi) ---")
    fig, axs = plt.subplots(K, P, figsize=(4 * P, 2 * K))
    
    for k in range(K):
        for p in range(P):
            # Gestion des cas spéciaux pour subplots 1x1, 1xP, Kx1
            if K == 1 and P == 1:
                ax = axs
            elif K == 1:
                ax = axs[p]
            elif P == 1:
                ax = axs[k]
            else:
                ax = axs[k, p]
                
            ax.plot(np.arange(L), Phi_cpu[k, :, p])
            ax.set_title(f'Atom {k} - chan {p}')
            ax.set_xlabel('Temps (L)')
            
    plt.tight_layout()
    plt.show()

def plot_reconstruction(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor, n_display: int = 1):
    """
    2) Affiche la reconstruction (X_recon) et l'original (X) pour quelques sujets,
       et calcule le MSE.
    
    Args:
        X (torch.Tensor): Tenseur des données originales de forme (S, N, P).
        Z (torch.Tensor): Tenseur des activations de forme (S, K, T).
        Phi (torch.Tensor): Tenseur des atomes de forme (K, L, P).
        n_display (int): Nombre de sujets à afficher.
    """
    X_cpu = X.detach().cpu()
    Z_cpu = Z.detach().cpu()
    Phi_cpu = Phi.detach().cpu()

    S, N, P = X_cpu.shape
    K, L, _ = Phi_cpu.shape
    
    print("\n--- 2) Reconstruction et MSE par sujet ---")
    n_display = min(n_display, S)
    
    # Prépare le tenseur Phi pour la convolution transposée
    phi_conv = Phi_cpu.permute(0, 2, 1)  # K x P x L
    
    for s in range(n_display):
        z_s = Z_cpu[s].unsqueeze(0)  # 1 x K x T
        
        # Calcul de la reconstruction: Conv Transpose
        # F.conv_transpose1d(input: 1xKxT, weight: KxPxL) -> 1xPx(T+L-1)
        recon = F.conv_transpose1d(z_s, phi_conv, padding=0).squeeze(0).permute(1, 0)  # N x P
        recon = recon.numpy()
        x_s = X_cpu[s].numpy()

        if recon.shape != x_s.shape:
            print(f"Warning shape mismatch for subject {s}: recon {recon.shape} vs x {x_s.shape}")
            continue # Passe au sujet suivant si la forme est incorrecte

        mse = ((x_s - recon) ** 2).mean()
        print(f'Subject {s} MSE: {mse:.6e}')

        for p in range(P):
            plt.figure(figsize=(12, 2.5))
            plt.plot(x_s[:, p], label='Original $X_s$')
            plt.plot(recon[:, p], label='Reconstruit $\hat{X}_s$', alpha=0.8)
            plt.title(f'Sujet {s} - Canal {p} (MSE={mse:.2e})')
            plt.legend()
            plt.xlabel('Temps (N)')
            plt.show()

def plot_activations(Z: torch.Tensor, n_display: int = 1):
    """
    3) Affiche la série temporelle des activations parcimonieuses Z (K x T).
    
    Args:
        Z (torch.Tensor): Tenseur des activations de forme (S, K, T).
        n_display (int): Nombre de sujets à afficher.
    """
    Z_cpu = Z.detach().cpu()
    S, K, T = Z_cpu.shape
    time_axis = np.arange(T)
    
    print("\n--- 3) Affichage des activations (Z) ---")
    n_display = min(n_display, S)

    for s in range(n_display):
        # Définir la grille de subplots (K lignes, 1 colonne)
        fig, axs = plt.subplots(K, 1, figsize=(12, 1.5 * K), sharex=True)
        
        if K == 1: # Gérer le cas K=1
            axs = [axs]
            
        for k in range(K):
            ax = axs[k]
            
            # Plot des activations pour l'atome k du sujet s
            activations_k = Z_cpu[s, k, :].numpy()
            
            # Utiliser la fonction plot pour les séries temporelles
            ax.plot(time_axis, activations_k, marker='.', linestyle='-', markersize=4, label=f'Atom {k}')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            
            ax.set_ylabel(f'Atom {k}', rotation=0, labelpad=30, fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

        fig.suptitle(f'Activations Parcimonieuses Z pour le Sujet {s} (K={K} atomes)', fontsize=14, y=1.02)
        axs[-1].set_xlabel('Temps T')
        
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

def print_metrics(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor):
    """
    4) Affiche la norme des atomes et 5) les statistiques globales de MSE.
    
    Args:
        X (torch.Tensor): Tenseur des données originales de forme (S, N, P).
        Z (torch.Tensor): Tenseur des activations de forme (S, K, T).
        Phi (torch.Tensor): Tenseur des atomes de forme (K, L, P).
    """
    X_cpu = X.detach().cpu()
    Z_cpu = Z.detach().cpu()
    Phi_cpu = Phi.detach().cpu()
    
    S, N, P = X_cpu.shape
    K, L, _ = Phi_cpu.shape
    
    # 4) Normes des atomes
    norms = Phi_cpu.view(K, -1).norm(p=2, dim=1).numpy()
    print("\n--- 4) Normes des atomes (Phi) ---")
    print('Normes L2 des atomes:', norms)
    
    # 5) Statistiques globales
    print("\n--- 5) Statistiques globales de MSE ---")
    
    mses = []
    phi_conv = Phi_cpu.permute(0, 2, 1)  # K x P x L
    
    for s in range(S):
        z_s = Z_cpu[s].unsqueeze(0)
        # Calcul de la reconstruction
        recon = F.conv_transpose1d(z_s, phi_conv, padding=0).squeeze(0).permute(1, 0).numpy()
        x_s = X_cpu[s].numpy()
        
        if recon.shape == x_s.shape:
            mses.append(((x_s - recon) ** 2).mean())

    if len(mses) > 0:
        print(f"MSE médiane sur {len(mses)} sujets: {np.median(mses):.6e}")
        print(f"MSE moyenne sur {len(mses)} sujets: {np.mean(mses):.6e}")
    else:
        print("Pas de MSE calculée (forme incorrecte pour tous les sujets).")

def full_plot_analysis(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor, n_display: int = 1):
    """
    Fonction principale pour exécuter l'analyse complète.
    
    Args:
        X (torch.Tensor): Tenseur des données originales.
        Z (torch.Tensor): Tenseur des activations.
        Phi (torch.Tensor): Tenseur des atomes.
        n_display (int): Nombre de sujets à afficher dans la reconstruction/activation.
    """
    # Exécuter les plots et métriques dans l'ordre
    plot_atoms(Phi)
    plot_reconstruction(X, Z, Phi, n_display)
    plot_activations(Z, n_display)
    print_metrics(X, Z, Phi)


def calculate_warped_reconstructions(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f, 
    s: int
):
    """
    Calcule la reconstruction originale et la reconstruction time-warped
    pour un sujet donné (s) et retourne les données CPU et les MSE.
    """
    device = X.device
    S, N, P = X.shape
    K, L, P_phi = Phi.shape
    
    if s >= S:
        raise ValueError(f"L'index du sujet {s} est hors limites (max {S-1}).")

    x_s = X[s].detach().cpu().numpy()  # (N, P)
    z_s = Z[s].unsqueeze(0).to(device)  # (1, K, T)
    a_s = A[s].to(device)               # (K, M)

    # 1. Reconstruction avec Phi original
    phi_conv = Phi.permute(0, 2, 1).to(device)  # (K, P, L)
    recon_orig = (
        F.conv_transpose1d(z_s, phi_conv, padding=0)
        .squeeze(0)
        .permute(1, 0)
        .detach()
        .cpu()
        .numpy()
    )  # (N, P)

    # 2. Reconstruction avec Phi "time-warped"
    warped_phis = []
    for k in range(K):
        phi_k = Phi[k].to(device)  # (L, P)
        a_k_s = a_s[k]             # (M,)
        # Appel de la fonction de warping
        warped_phi_k = time_warping_f(phi_k, a_k_s)  # (L, P) 
        
        if not isinstance(warped_phi_k, torch.Tensor):
             raise TypeError("time_warping_f doit retourner un torch.Tensor.")
        
        warped_phis.append(warped_phi_k.unsqueeze(0))
        
    Phi_warped = torch.cat(warped_phis, dim=0)  # (K, L, P)
    
    phi_conv_warped = Phi_warped.permute(0, 2, 1)
    recon_warped = (
        F.conv_transpose1d(z_s, phi_conv_warped, padding=0)
        .squeeze(0)
        .permute(1, 0)
        .detach()
        .cpu()
        .numpy()
    )  # (N, P)

    # 3. Calcul du MSE
    if recon_orig.shape != x_s.shape or recon_warped.shape != x_s.shape:
        print("Avertissement: Les formes de reconstruction ne correspondent pas à X. Skip MSE.")
        P_val = x_s.shape[1] 
        return x_s, recon_orig, recon_warped, np.zeros(P_val), np.zeros(P_val)

    mse_per_channel_orig = ((x_s - recon_orig) ** 2).mean(axis=0)  # (P,)
    mse_per_channel_warped = ((x_s - recon_warped) ** 2).mean(axis=0)  # (P,)

    return x_s, recon_orig, recon_warped, mse_per_channel_orig, mse_per_channel_warped

def plot_warped_reconstruction(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f, 
    s: int = 0
):
    """
    Affiche l'original et les deux reconstructions (standard et time-warped)
    pour chaque canal d'un sujet spécifié (s).
    """
    print(f"\n--- 1) Visualisation de la Reconstruction (Sujet {s}) ---")
    
    try:
        x_s, recon_orig, recon_warped, mse_orig, mse_warped = \
            calculate_warped_reconstructions(X, Z, Phi, A, time_warping_f, s)
    except Exception as e:
        print(f"Erreur lors du calcul des reconstructions: {e}")
        return

    P = x_s.shape[1]
    
    for p in range(P):
        plt.figure(figsize=(12, 3.5))
        plt.plot(x_s[:, p], label='Original $X_s$', linewidth=1.5, color='black')
        plt.plot(recon_orig[:, p], label='Recon (Φ original)', alpha=0.7, linestyle='--')
        plt.plot(recon_warped[:, p], label='Recon (Φ time-warped)', alpha=0.7, linestyle='-')
        
        plt.title(
            f'Sujet {s} - Canal {p}\n'
            f'MSE original={mse_orig[p]:.4e} | MSE warped={mse_warped[p]:.4e}'
        )
        plt.xlabel('Temps (N)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

def plot_mse_comparison(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f, 
    s: int = 0
):
    """
    Affiche un barplot comparant le MSE par canal pour la reconstruction
    standard (Phi) et la reconstruction time-warped (Phi_warped).
    """
    print(f"\n--- 2) Barplot de Comparaison des MSE (Sujet {s}) ---")
    
    try:
        _, _, _, mse_orig, mse_warped = \
            calculate_warped_reconstructions(X, Z, Phi, A, time_warping_f, s)
    except Exception as e:
        print(f"Erreur lors du calcul des MSE: {e}")
        return

    P = len(mse_orig)
    x_axis = np.arange(P)
    bar_width = 0.35
    
    plt.figure(figsize=(9, 5))
    
    plt.bar(
        x_axis - bar_width/2, 
        mse_orig, 
        width=bar_width, 
        label='Φ original', 
        color='skyblue', 
        edgecolor='black'
    )
    plt.bar(
        x_axis + bar_width/2, 
        mse_warped, 
        width=bar_width, 
        label='Φ time-warped', 
        color='lightcoral', 
        edgecolor='black'
    )
    
    plt.xlabel('Canal (P)')
    plt.ylabel('Erreur Quadratique Moyenne (MSE)')
    plt.title(f'Comparaison du MSE par Canal - Sujet {s}')
    plt.xticks(x_axis, [f'Canal {p}' for p in range(P)])
    plt.yscale('log') # Utilisation de l'échelle log pour mieux visualiser les petites valeurs
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def print_warping_metrics(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f, 
    s: int = 0
):
    """
    Calcule et imprime les MSE par canal et globales pour les deux reconstructions.
    """
    print(f"\n--- 3) Métriques de Reconstruction (Sujet {s}) ---")
    
    try:
        _, _, _, mse_orig, mse_warped = \
            calculate_warped_reconstructions(X, Z, Phi, A, time_warping_f, s)
    except Exception as e:
        print(f"Erreur lors du calcul des métriques: {e}")
        return

    P = len(mse_orig)
    
    print("\n📊 MSE par canal (Comparaison):")
    for p in range(P):
        print(f"  Canal {p}: Original={mse_orig[p]:.4e}, Warped={mse_warped[p]:.4e}")

    # MSE Globale pour le sujet s
    mse_global_orig = np.mean(mse_orig)
    mse_global_warped = np.mean(mse_warped)
    
    print("\n📈 MSE Globale:")
    print(f"  MSE Globale (Φ original): {mse_global_orig:.4e}")
    print(f"  MSE Globale (Φ time-warped): {mse_global_warped:.4e}")
    
    if mse_global_warped < mse_global_orig:
        gain = ((mse_global_orig - mse_global_warped) / mse_global_orig) * 100
        print(f"  Gain de réduction d'erreur: {gain:.2f} %")
    elif mse_global_orig == 0:
        print("  MSE originale est zéro, impossible de calculer le gain.")
    else:
        # Note: Si le MSE warped est plus grand, ce n'est pas un gain, mais une perte
        loss = ((mse_global_warped - mse_global_orig) / mse_global_orig) * 100
        print(f"  Perte de performance: {loss:.2f} %")


def full_warping_analysis(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f=time_warping_f, 
    s: int = 0
):
    """
    Exécute l'analyse complète de la reconstruction avec time-warping
    pour un sujet donné.
    """
    print(f"\n===========================================================")
    print(f"  Analyse Complète Time-Warping - Sujet {s} / {X.shape[0]}")
    print(f"===========================================================")
    
    print_warping_metrics(X, Z, Phi, A, time_warping_f, s)
    plot_warped_reconstruction(X, Z, Phi, A, time_warping_f, s)
    plot_mse_comparison(X, Z, Phi, A, time_warping_f, s)