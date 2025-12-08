import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    1) Displays atoms (Phi) per channel.
    
    Args:
        Phi (torch.Tensor): Atoms tensor of shape (K, L, P).
    """
    Phi_cpu = Phi.detach().cpu()
    K, L, P = Phi_cpu.shape
    
    print("--- Displaying Atoms (Phi) ---")
    fig, axs = plt.subplots(K, P, figsize=(4 * P, 2 * K))
    
    for k in range(K):
        for p in range(P):
            # Handling special cases for 1x1, 1xP, Kx1 subplots
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
            ax.set_xlabel('Length (L)')
            
    plt.tight_layout()
    plt.show()

def plot_reconstruction(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor, n_display: int = 1):
    """
    Displays reconstruction + original signal, with colored bars indicating 
    activations (start time + color according to the atom).
    If multiple atoms are activated at the same time t:
        --> only the atom with the maximum activation is displayed.
    
    Args:
        X (torch.Tensor): Original data tensor.
        Z (torch.Tensor): Activations tensor.
        Phi (torch.Tensor): Atoms tensor.
        n_display (int): Number of subjects to display.
    """

    # --- Move to CPU ---
    X_cpu = X.detach().cpu()
    Z_cpu = Z.detach().cpu()
    Phi_cpu = Phi.detach().cpu()

    S, N, P = X_cpu.shape
    K, L, _ = Phi_cpu.shape
    T = Z_cpu.shape[-1]

    print("\n--- Reconstruction and MSE per Subject ---")
    n_display = min(n_display, S)

    phi_conv = Phi_cpu.permute(0, 2, 1) 

    atom_colors = {0: "red", 1: "blue"} 

    for s in range(n_display):
        z_s = Z_cpu[s].unsqueeze(0) 

        recon = F.conv_transpose1d(z_s, phi_conv).squeeze(0).permute(1, 0).numpy()
        x_s = X_cpu[s].numpy() # (N, P)

        
        activations = []
        for t in range(T):
            z_t = Z_cpu[s, :, t]        
            k_max = torch.argmax(z_t)    
            val_max = z_t[k_max].item()

            if val_max > 0:              
                activations.append((int(k_max), int(t)))

        for p in range(P):
            
            mse_channel = float(((x_s[:, p] - recon[:, p]) ** 2).mean())
            
            fig, ax = plt.subplots(figsize=(12, 3))

            y_all_min = min(np.min(x_s[:, p]), np.min(recon[:, p]))
            y_all_max = max(np.max(x_s[:, p]), np.max(recon[:, p]))
            rng = max(y_all_max - y_all_min, 1.0)
            margin = 0.05 * rng
            y_bottom = y_all_min - margin
            y_top = y_all_max + margin

            for (k, t) in activations:
                start = t
                end = min(t + L, N) 
                color = atom_colors.get(k, "black")

                rect = patches.Rectangle(
                    (start, y_bottom),
                    end - start,
                    (y_top - y_bottom),
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=0
                )
                ax.add_patch(rect)

                seg_height = 0.08 * (y_top - y_bottom)
                ax.plot([start, start],
                        [y_top - seg_height, y_top],
                        color=color,
                        linewidth=2,
                        zorder=2)

            ax.plot(x_s[:, p], label='Original', linewidth=1.5, zorder=3)
            ax.plot(recon[:, p], label='Reconstructed', linewidth=1.25, alpha=0.9, zorder=4)

            ax.set_ylim(y_bottom, y_top)
            ax.set_title(f"Subject {s} - Channel {p}   (MSE={mse_channel:.2e})")
            ax.set_xlabel("Time (N)")
            ax.legend()

            plt.show()



def plot_activations(Z: torch.Tensor, n_display: int = 1):
    """
    3) Displays the time series of sparse activations Z (K x T).
    
    Args:
        Z (torch.Tensor): Activations tensor of shape (S, K, T).
        n_display (int): Number of subjects to display.
    """
    Z_cpu = Z.detach().cpu()
    S, K, T = Z_cpu.shape
    time_axis = np.arange(T)
    
    print("\n--- Displaying Activations (Z) ---")
    n_display = min(n_display, S)

    for s in range(n_display):
        fig, axs = plt.subplots(K, 1, figsize=(12, 1.5 * K), sharex=True)
        
        if K == 1: 
            axs = [axs]
            
        for k in range(K):
            ax = axs[k]
            
            activations_k = Z_cpu[s, k, :].numpy()
            
            ax.plot(time_axis, activations_k, marker='.', linestyle='-', markersize=4, label=f'Atom {k}')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            
            ax.set_ylabel(f'Atom {k}', rotation=0, labelpad=30, fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

        fig.suptitle(f'Sparse Activations Z for Subject {s} (K={K} atoms)', fontsize=14, y=1.02)
        
        axs[-1].set_xlabel('Time T')
        
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

def print_metrics(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor):
    """
    Prints global MSE statistics.
    Prints average sparsity metrics.
    
    Args:
        X (torch.Tensor): Original data tensor of shape (S, N, P).
        Z (torch.Tensor): Activations tensor of shape (S, K, T).
        Phi (torch.Tensor): Atoms tensor of shape (K, L, P).
    """
    X_cpu = X.detach().cpu()
    Z_cpu = Z.detach().cpu()
    Phi_cpu = Phi.detach().cpu()
    
    S, N, P = X_cpu.shape
    K, L, _ = Phi_cpu.shape
    

    # --- 5) Global MSE Statistics ---
    print("\n--- Global Mean Squared Error (MSE) Statistics ---")
    
    mses = []
    # Permutation for F.conv_transpose1d: (K, P, L)
    phi_conv = Phi_cpu.permute(0, 2, 1)  
    
    for s in range(S):
        z_s = Z_cpu[s].unsqueeze(0)
        # Calculate reconstruction
        recon = F.conv_transpose1d(z_s, phi_conv, padding=0).squeeze(0).permute(1, 0).numpy()
        x_s = X_cpu[s].numpy()
        
        # Check if dimensions match
        if recon.shape == x_s.shape:
            mses.append(((x_s - recon) ** 2).mean())

    if len(mses) > 0:
        mses_np = np.array(mses)
        print(f"Subjects analyzed: {len(mses)}")
        print(f"MSE Mean: {np.mean(mses_np):.6e}")
        print(f"MSE Median: {np.median(mses_np):.6e}")
        print(f"MSE Minimum: {np.min(mses_np):.6e}")
        print(f"MSE Maximum: {np.max(mses_np):.6e}")
        print(f"MSE Standard Deviation: {np.std(mses_np):.6e}")
    else:
        print("No MSE calculated (incorrect shape for all subjects).")

    # --- 6) Sparsity Statistics ---
    # Sparsity is the number of non-zero Z elements / total number of elements in Z.
    # Using a small threshold (1e-10) to account for potential floating-point errors.
    
    # Counts elements > 0
    sparsity_level = (Z_cpu > 1e-10).sum().item() / Z_cpu.numel()
    
    # Average number of non-zero activations per subject
    avg_active_coefficients = (Z_cpu > 1e-10).sum().item() / S
    
    print("\n--- Sparsity Statistics ---")
    print(f"Sparsity Rate (coeffs > 0): {sparsity_level:.4%}")
    print(f"Avg. number of active coefficients per subject: {avg_active_coefficients:.2f}")

def full_plot_analysis(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor, n_display: int = 1):
    """
    Main function to execute the complete analysis.
    
    Args:
        X (torch.Tensor): Original data tensor.
        Z (torch.Tensor): Activations tensor.
        Phi (torch.Tensor): Atoms tensor.
        n_display (int): Number of subjects to display in the reconstruction/activation plots.
    """
    plot_atoms(Phi)
    plot_reconstruction(X, Z, Phi, n_display)
    plot_activations(Z, n_display)
    print_metrics(X, Z, Phi)


# ==============================================================================
# UTILITY FUNCTION (Calculation Core)
# ==============================================================================

def calculate_warped_reconstructions(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f, 
    s: int
):
    """
    Calculates the standard (original Phi) and time-warped reconstructions 
    for a given subject (s) and returns CPU data and channel-wise MSEs.
    """
    device = X.device

    X = X.float()
    Z = Z.float()
    Phi = Phi.float()
    A = A.float()
    
    try:
        S, N, P = X.shape
        K, L, P_phi = Phi.shape
    except:
        S,N = X.shape
        K,L = Phi.shape
        P = 1
    
    if s >= S:
        raise ValueError(f"Subject index {s} is out of bounds (max {S-1}).")

    x_s = X[s].detach().cpu().numpy()  # (N, P)
    z_s = Z[s].unsqueeze(0).to(device)  # (1, K, T)
    a_s = A[s].to(device)               # (K, M)

    # 1. Standard Reconstruction (Original Phi)
    phi_conv = Phi.permute(0, 2, 1).to(device)  # (K, P, L)
    recon_orig = (
        F.conv_transpose1d(z_s, phi_conv, padding=0)
        .squeeze(0)
        .permute(1, 0)
        .detach()
        .cpu()
        .numpy()
    )  # (N, P)

    # 2. Time-Warped Reconstruction
    warped_phis = []
    for k in range(K):
        phi_k = Phi[k].to(device)  # (L, P)
        a_k_s = a_s[k]             # (M,)
        # Call the warping function
        warped_phi_k = time_warping_f(phi_k, a_k_s)  # (L, P) 
        
        if not isinstance(warped_phi_k, torch.Tensor):
             raise TypeError("time_warping_f must return a torch.Tensor.")
        
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

    # 3. Calculate MSE
    if recon_orig.shape != x_s.shape or recon_warped.shape != x_s.shape:
        print("Warning: Reconstruction shapes do not match X. Skipping MSE calculation.")
        P_val = x_s.shape[1] 
        # Return zeros for MSE if shapes are incorrect
        return x_s, recon_orig, recon_warped, np.zeros(P_val), np.zeros(P_val)

    # Calculate MSE per channel
    mse_per_channel_orig = ((x_s - recon_orig) ** 2).mean(axis=0)  # (P,)
    mse_per_channel_warped = ((x_s - recon_warped) ** 2).mean(axis=0)  # (P,)

    return x_s, recon_orig, recon_warped, mse_per_channel_orig, mse_per_channel_warped

# ==============================================================================
# PLOT FUNCTIONS
# ==============================================================================

def plot_warped_reconstruction(
    x_s: np.ndarray, 
    recon_orig: np.ndarray, 
    recon_warped: np.ndarray, 
    mse_orig: np.ndarray, 
    mse_warped: np.ndarray, 
    s: int
):
    """
    Displays the original signal and both reconstructions (standard and time-warped)
    for each channel of a specified subject (s).
    """
    print(f"\n--- Reconstruction Visualization (Subject {s}) ---")
    
    P = x_s.shape[1]
    
    for p in range(P):
        plt.figure(figsize=(12, 3.5))
        plt.plot(x_s[:, p], label='Original $X_s$', linewidth=1.5, color='black')
        plt.plot(recon_orig[:, p], label='Recon (Original Φ)', alpha=0.7, linestyle='--')
        plt.plot(recon_warped[:, p], label='Recon (Time-Warped Φ)', alpha=0.7, linestyle='-')
        
        plt.title(
            f'Subject {s} - Channel {p}\n'
            f'MSE Original={mse_orig[p]:.4e} | MSE Warped={mse_warped[p]:.4e}'
        )
        plt.xlabel('Time (N)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

def plot_mse_comparison(mse_orig: np.ndarray, mse_warped: np.ndarray, s: int):
    """
    Displays a bar plot comparing the MSE per channel for standard 
    and time-warped reconstruction.
    """
    print(f"\n--- MSE Comparison Barplot (Subject {s}) ---")
    
    P = len(mse_orig)
    x_axis = np.arange(P)
    bar_width = 0.35
    
    plt.figure(figsize=(9, 5))
    
    plt.bar(
        x_axis - bar_width/2, 
        mse_orig, 
        width=bar_width, 
        label='Original Φ', 
        color='skyblue', 
        edgecolor='black'
    )
    plt.bar(
        x_axis + bar_width/2, 
        mse_warped, 
        width=bar_width, 
        label='Time-Warped Φ', 
        color='lightcoral', 
        edgecolor='black'
    )
    
    plt.xlabel('Channel (P)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'MSE Comparison per Channel - Subject {s}')
    plt.xticks(x_axis, [f'Channel {p}' for p in range(P)])
    plt.yscale('log') # Use log scale for better visualization of small values
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_personalized_vs_general_atoms_by_channel(Phi: torch.Tensor, A: torch.Tensor, time_warping_f, s: int):
    """
    Displays the general and personalized (time-warped) atoms for each channel
    of a given subject.
    """
    print(f"\n--- General vs. Personalized Atoms (Subject {s}) ---")

    K, L, P = Phi.shape

    for p in range(P):
        fig, axes = plt.subplots(K, 1, figsize=(12, 2.5*K), sharex=True)
        if K == 1:
            axes = [axes]  # ensure it is always a list

        for k, ax in enumerate(axes):
            phi_k = Phi[k]  # (L, P)
            a_k_s = A[s, k]
            
            # Warping calculation must handle device transfer if needed
            warped_phi_k = time_warping_f(phi_k.to(A.device), a_k_s.to(A.device)).cpu()

            ax.plot(phi_k[:, p].cpu().numpy(), '--', label=f'General - Atom {k}', alpha=0.7)
            ax.plot(warped_phi_k[:, p].numpy(), '-', label=f'Personalized Subject {s} - Atom {k}', alpha=0.7)
            
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)

        axes[-1].set_xlabel('Time (L)')
        fig.suptitle(f'Channel {p} : General vs. Personalized Atoms (Subject {s})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def full_warping_analysis(
    X: torch.Tensor, 
    Z: torch.Tensor, 
    Phi: torch.Tensor, 
    A: torch.Tensor, 
    time_warping_f=time_warping_f, 
    s: int = 0
):
    """
    Executes the complete reconstruction analysis with time-warping
    for a given subject (s).
    
    Args:
        X (torch.Tensor): Original data tensor.
        Z (torch.Tensor): Activations tensor.
        Phi (torch.Tensor): General atoms tensor.
        A (torch.Tensor): Personalized parameters tensor (e.g., warping coeffs).
        time_warping_f: The warping function (e.g., time_warping_f(phi_k, a_k_s)).
        s (int): Index of the subject to analyze and display.
    """
    
    S = X.shape[0]

    # --- Initial Printout ---
    print(f"\n===========================================================")
    print(f"  FULL TIME-WARPING ANALYSIS - Subject {s} of {S}")
    print(f"===========================================================")

    try:
        # Calculate reconstructions and MSEs once
        x_s, recon_orig, recon_warped, mse_orig, mse_warped = \
            calculate_warped_reconstructions(X, Z, Phi, A, time_warping_f, s)
        
    except Exception as e:
        print(f"\nFATAL ERROR during reconstruction calculation for Subject {s}: {e}")
        return

    # --- 4) Print Metrics ---
    
    print(f"\n--- Reconstruction Metrics (Subject {s}) ---")
    
    P = len(mse_orig)
    
    print("\n MSE per Channel (Comparison):")
    for p in range(P):
        print(f"  Channel {p}: Original={mse_orig[p]:.4e}, Warped={mse_warped[p]:.4e}")

    # Global MSE for subject s
    mse_global_orig = np.mean(mse_orig)
    mse_global_warped = np.mean(mse_warped)
    
    print("\n Global MSE:")
    print(f"  Global MSE (Original Φ): {mse_global_orig:.4e}")
    print(f"  Global MSE (Time-Warped Φ): {mse_global_warped:.4e}")
    
    if mse_global_warped < mse_global_orig:
        gain = ((mse_global_orig - mse_global_warped) / mse_global_orig) * 100
        print(f"  Error Reduction Gain: {gain:.2f} %")
    elif mse_global_orig == 0:
        print("  Original MSE is zero, cannot calculate gain.")
    else:
        # Note: If warped MSE is larger, it's a loss
        loss = ((mse_global_warped - mse_global_orig) / mse_global_orig) * 100
        print(f"  Performance Loss: {loss:.2f} %")


    # --- 1) Plot Reconstruction ---
    plot_warped_reconstruction(x_s, recon_orig, recon_warped, mse_orig, mse_warped, s)
    
    # --- 2) Plot MSE Comparison ---
    plot_mse_comparison(mse_orig, mse_warped, s)
    
    # --- 3) Plot Atoms Comparison ---
    plot_personalized_vs_general_atoms_by_channel(Phi, A, time_warping_f, s)