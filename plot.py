import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from tools import time_warping_f

# ==============================================================================
# UTILS
# ==============================================================================

def check_tensors(X, Z, Phi):
    """
    Checks the existence of tensors, moves them to CPU, and checks for NaNs.
    """
    if X is None or Z is None or Phi is None:
        raise ValueError("Tensors Phi, Z, and X must not be None.")

    # Move to CPU for visualization
    Phi_cpu = Phi.detach().cpu()
    Z_cpu = Z.detach().cpu()
    X_cpu = X.detach().cpu()

    # Quick NaN check
    if torch.isnan(Phi_cpu).any() or torch.isnan(Z_cpu).any() or torch.isnan(X_cpu).any():
        print("Warning: NaN detected in Phi, Z, or X!")
        
    return X_cpu, Z_cpu, Phi_cpu

def get_atom_color(k, total_k):
    """Returns a consistent color for atom k."""
    cmap = plt.get_cmap('tab10' if total_k <= 10 else 'tab20')
    return cmap(k % 20)

# ==============================================================================
# BASIC PLOTTING FUNCTIONS
# ==============================================================================

def plot_atoms(Phi: torch.Tensor):
    """
    Displays atoms (Phi) per channel.
    Generalized to handle any (K, L, P) shape using squeeze=False.
    """
    Phi_cpu = Phi.detach().cpu()
    K, L, P = Phi_cpu.shape
    
    print("--- Displaying Atoms (Phi) ---")
    
    # squeeze=False ensures axs is always a 2D array [K, P]
    fig, axs = plt.subplots(K, P, figsize=(3 * P, 2 * K), squeeze=False, sharex=True)
    
    for k in range(K):
        for p in range(P):
            ax = axs[k, p]
            ax.plot(np.arange(L), Phi_cpu[k, :, p], color=get_atom_color(k, K))
            
            # Generic titling
            if k == 0:
                ax.set_title(f'Channel {p}')
            if p == 0:
                ax.set_ylabel(f'Atom {k}')
                
            ax.grid(True, linestyle=':', alpha=0.5)
            
    # Set common xlabel at the bottom
    for ax in axs[-1, :]:
        ax.set_xlabel('Time (L)')
            
    plt.tight_layout()
    plt.show()

def plot_reconstruction(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor, n_display: int = 1):
    """
    Displays reconstruction + original signal with colored activation bars.
    Works for any number of channels P and atoms K.
    """
    X_cpu, Z_cpu, Phi_cpu = check_tensors(X, Z, Phi)

    S, N, P = X_cpu.shape
    K, L, _ = Phi_cpu.shape
    T = Z_cpu.shape[-1]

    print("\n--- Reconstruction and MSE per Subject ---")
    n_display = min(n_display, S)

    phi_conv = Phi_cpu.permute(0, 2, 1) # (K, P, L) for conv1d

    for s in range(n_display):
        z_s = Z_cpu[s].unsqueeze(0) 

        # Reconstruction
        recon = F.conv_transpose1d(z_s, phi_conv).squeeze(0).permute(1, 0).numpy()
        x_s = X_cpu[s].numpy() # (N, P)

        # Collect activations for visualization
        # List of tuples: (atom_index, time_index, value)
        activations = []
        for t in range(T):
            z_t = Z_cpu[s, :, t]        
            k_max = torch.argmax(z_t)    
            val_max = z_t[k_max].item()

            if val_max > 0:              
                activations.append((int(k_max), int(t)))

        # Plot per channel
        for p in range(P):
            if recon.shape[0] != x_s.shape[0]:
                print(f"Shape mismatch: X={x_s.shape}, Recon={recon.shape}. Skipping plot.")
                continue

            mse_channel = float(((x_s[:, p] - recon[:, p]) ** 2).mean())
            
            fig, ax = plt.subplots(figsize=(12, 3))

            # Dynamic y-limits
            y_all_min = min(np.min(x_s[:, p]), np.min(recon[:, p]))
            y_all_max = max(np.max(x_s[:, p]), np.max(recon[:, p]))
            rng = max(y_all_max - y_all_min, 1e-5)
            margin = 0.1 * rng
            y_bottom = y_all_min - margin
            y_top = y_all_max + margin

            # Draw activation rectangles
            for (k, t) in activations:
                start = t
                end = min(t + L, N) 
                color = get_atom_color(k, K)

                # Colored background box
                rect = patches.Rectangle(
                    (start, y_bottom),
                    end - start,
                    (y_top - y_bottom),
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=0
                )
                ax.add_patch(rect)

                # Colored marker at the top
                seg_height = 0.05 * (y_top - y_bottom)
                ax.plot([start, start],
                        [y_top - seg_height, y_top],
                        color=color,
                        linewidth=2,
                        zorder=2)

            # Signal plots
            ax.plot(x_s[:, p], label='Original', linewidth=1.5, color='black', alpha=0.7, zorder=3)
            ax.plot(recon[:, p], label='Reconstructed', linewidth=1.25, linestyle='--', color='darkorange', zorder=4)

            ax.set_ylim(y_bottom, y_top)
            ax.set_title(f"Subject {s} - Channel {p} (MSE={mse_channel:.2e})")
            ax.set_xlabel("Time (N)")
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            plt.show()

def plot_activations(Z: torch.Tensor, n_display: int = 1):
    """
    Displays the time series of sparse activations Z (K x T).
    """
    Z_cpu = Z.detach().cpu()
    S, K, T = Z_cpu.shape
    time_axis = np.arange(T)
    
    print("\n--- Displaying Activations (Z) ---")
    n_display = min(n_display, S)

    for s in range(n_display):
        # squeeze=False ensures axs is always iterable
        fig, axs = plt.subplots(K, 1, figsize=(12, 1.5 * K), sharex=True, squeeze=False)
            
        for k in range(K):
            ax = axs[k, 0] # Indexing into (rows, 1) array
            
            activations_k = Z_cpu[s, k, :].numpy()
            color = get_atom_color(k, K)
            
            ax.plot(time_axis, activations_k, marker='.', linestyle='-', markersize=4, label=f'Atom {k}', color=color)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            
            ax.set_ylabel(f'Atom {k}', rotation=0, labelpad=30, fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

        fig.suptitle(f'Sparse Activations Z for Subject {s} (K={K})', fontsize=14, y=1.00)
        axs[-1, 0].set_xlabel('Time T')
        
        plt.tight_layout()
        plt.show()

def print_metrics(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor):
    """
    Prints global MSE and Sparsity statistics.
    """
    X_cpu, Z_cpu, Phi_cpu = check_tensors(X, Z, Phi)
    S = X_cpu.shape[0]

    print("\n--- Global Statistics ---")
    
    mses = []
    phi_conv = Phi_cpu.permute(0, 2, 1)  
    
    for s in range(S):
        z_s = Z_cpu[s].unsqueeze(0)
        recon = F.conv_transpose1d(z_s, phi_conv, padding=0).squeeze(0).permute(1, 0).numpy()
        x_s = X_cpu[s].numpy()
        
        if recon.shape == x_s.shape:
            mses.append(((x_s - recon) ** 2).mean())

    if len(mses) > 0:
        mses_np = np.array(mses)
        print(f"MSE Mean: {np.mean(mses_np):.6e} | Std: {np.std(mses_np):.6e}")
        print(f"MSE Min : {np.min(mses_np):.6e} | Max: {np.max(mses_np):.6e}")
    else:
        print("No MSE calculated (incorrect shapes).")

    # Sparsity
    sparsity_rate = (Z_cpu > 1e-9).float().mean().item()
    avg_active = (Z_cpu > 1e-9).float().sum().item() / S
    
    print(f"Sparsity Rate (>1e-9): {sparsity_rate:.4%}")
    print(f"Avg active coefficients per subject: {avg_active:.2f}")

def full_plot_analysis(X: torch.Tensor, Z: torch.Tensor, Phi: torch.Tensor, n_display: int = 1):
    plot_atoms(Phi)
    plot_reconstruction(X, Z, Phi, n_display)
    plot_activations(Z, n_display)
    print_metrics(X, Z, Phi)


# ==============================================================================
# TIME WARPING UTILS & ANALYSIS
# ==============================================================================

def calculate_warped_reconstructions(X, Z, Phi, A, time_warping_f, s):
    """
    Calculates standard vs warped reconstructions for subject s.
    """
    device = X.device
    S, N, P = X.shape
    K, L, P_phi = Phi.shape
    
    if s >= S:
        raise ValueError(f"Subject index {s} out of bounds.")

    x_s = X[s].detach().cpu().numpy()
    z_s = Z[s].unsqueeze(0).to(device)
    a_s = A[s].to(device)

    # 1. Standard
    phi_conv = Phi.permute(0, 2, 1).to(device)
    recon_orig = F.conv_transpose1d(z_s, phi_conv).squeeze(0).permute(1, 0).detach().cpu().numpy()

    # 2. Warped
    warped_phis = []
    for k in range(K):
        phi_k = Phi[k].to(device)
        a_k_s = a_s[k]
        warped_phi_k = time_warping_f(phi_k, a_k_s)
        warped_phis.append(warped_phi_k.unsqueeze(0))
        
    Phi_warped = torch.cat(warped_phis, dim=0)
    phi_conv_warped = Phi_warped.permute(0, 2, 1)
    recon_warped = F.conv_transpose1d(z_s, phi_conv_warped).squeeze(0).permute(1, 0).detach().cpu().numpy()

    if recon_orig.shape != x_s.shape or recon_warped.shape != x_s.shape:
        return x_s, recon_orig, recon_warped, np.zeros(P), np.zeros(P)

    mse_per_channel_orig = ((x_s - recon_orig) ** 2).mean(axis=0)
    mse_per_channel_warped = ((x_s - recon_warped) ** 2).mean(axis=0)

    return x_s, recon_orig, recon_warped, mse_per_channel_orig, mse_per_channel_warped

def plot_warped_reconstruction(x_s, recon_orig, recon_warped, mse_orig, mse_warped, s):
    """
    Visualizes original vs warped reconstruction per channel.
    """
    print(f"\n--- Warped Reconstruction Visualization (Subject {s}) ---")
    P = x_s.shape[1]
    
    for p in range(P):
        plt.figure(figsize=(12, 3.5))
        plt.plot(x_s[:, p], label='Original Signal', linewidth=1.5, color='black', alpha=0.8)
        plt.plot(recon_orig[:, p], label='Recon (General Φ)', alpha=0.7, linestyle='--', color='tab:blue')
        plt.plot(recon_warped[:, p], label='Recon (Warped Φ)', alpha=0.7, linestyle='-', color='tab:red')
        
        plt.title(f'Subject {s} - Channel {p} | MSE Orig: {mse_orig[p]:.2e} | MSE Warped: {mse_warped[p]:.2e}')
        plt.xlabel('Time (N)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

def plot_mse_comparison(mse_orig, mse_warped, s):
    """
    Barplot comparing MSEs.
    """
    print(f"\n--- MSE Comparison (Subject {s}) ---")
    P = len(mse_orig)
    x_axis = np.arange(P)
    width = 0.35
    
    plt.figure(figsize=(max(6, P), 5))
    plt.bar(x_axis - width/2, mse_orig, width=width, label='General Φ', color='skyblue', edgecolor='black')
    plt.bar(x_axis + width/2, mse_warped, width=width, label='Warped Φ', color='lightcoral', edgecolor='black')
    
    plt.xlabel('Channel Index')
    plt.ylabel('MSE (Log Scale)')
    plt.title(f'MSE Comparison per Channel - Subject {s}')
    plt.xticks(x_axis, [f'Ch {p}' for p in range(P)])
    plt.yscale('log')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_personalized_vs_general_atoms_by_channel(Phi, A, time_warping_f, s):
    """
    Plots General vs Personalized atoms for Subject s.
    """
    print(f"\n--- General vs. Personalized Atoms (Subject {s}) ---")
    
    K, L, P = Phi.shape
    
    # Iterate over channels
    for p in range(P):
        # Create subplots for all atoms in this channel
        # squeeze=False to ensure it's iterable
        fig, axs = plt.subplots(K, 1, figsize=(10, 2.0 * K), sharex=True, squeeze=False)
        
        for k in range(K):
            ax = axs[k, 0]
            phi_k = Phi[k] # (L, P)
            a_k_s = A[s, k]
            
            # Warp
            warped_phi_k = time_warping_f(phi_k.to(A.device), a_k_s.to(A.device)).detach().cpu()
            phi_k_cpu = phi_k.detach().cpu()
            
            color = get_atom_color(k, K)
            ax.plot(phi_k_cpu[:, p], '--', label=f'General (Atom {k})', color='gray', alpha=0.6)
            ax.plot(warped_phi_k[:, p], '-', label=f'Personalized (Subj {s})', color=color, alpha=0.9)
            
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.5)
            
            if k == 0:
                ax.set_title(f"Channel {p} - Atom Variations")
        
        axs[-1, 0].set_xlabel('Time (L)')
        plt.tight_layout()
        plt.show()

def full_warping_analysis(X, Z, Phi, A, time_warping_f=time_warping_f, s=0):
    """
    Master function for warping analysis.
    """
    S = X.shape[0]
    print(f"\n{'='*60}\n  FULL TIME-WARPING ANALYSIS - Subject {s}/{S}\n{'='*60}")

    try:
        x_s, recon_orig, recon_warped, mse_orig, mse_warped = \
            calculate_warped_reconstructions(X, Z, Phi, A, time_warping_f, s)
    except Exception as e:
        print(f"Error calculating reconstructions: {e}")
        return

    # Print Gains
    mse_global_orig = np.mean(mse_orig)
    mse_global_warped = np.mean(mse_warped)
    
    print(f"\n Global MSE (General): {mse_global_orig:.4e}")
    print(f" Global MSE (Warped) : {mse_global_warped:.4e}")
    
    if mse_global_orig > 0:
        gain = ((mse_global_orig - mse_global_warped) / mse_global_orig) * 100
        print(f" Improvement Gain    : {gain:.2f}%")

    plot_warped_reconstruction(x_s, recon_orig, recon_warped, mse_orig, mse_warped, s)
    plot_mse_comparison(mse_orig, mse_warped, s)
    plot_personalized_vs_general_atoms_by_channel(Phi, A, time_warping_f, s)