from percdl_federated.__init__ import *
from data import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from click.testing import CliRunner
import click
from tools import find_optimal_cutoff, apply_low_pass
import tools
import torch
import plot


def plot_results(X,Z,A,Phi,function,save_path=None):
    return plot.full_warping_analysis(X,Z,Phi,A)


def test_percdl_federated(X,function,phi_init,K,L,n_steps=200,step_size=1e-3,save_path=None):

    print("Starting PerCDL federated test...")
    assert X.ndim == 2, "X should be 2D"
    X_numpy = np.array(X)
    X_jax = jnp.array(X_numpy)
    model = PerCDL(function,X_jax,K,L,n_steps=n_steps,step_size=step_size)
    model.initialize(Phi=phi_init)
    print("Initialized PerCDL federated model")
    Z = model.Z
    A = model.A
    Phi = model.Phi

    model.run()
    print("Ran PerCDL federated model")
    if save_path:
        plot_results(torch.tensor(X[..., None]), torch.tensor(model.Z), torch.tensor(model.A), torch.tensor(model.Phi[...,None]), function, save_path=save_path)
        print(f"Saved results to {save_path}")
    else:
        plot_results(torch.tensor(X[..., None]), torch.tensor(model.Z), torch.tensor(model.A), torch.tensor(model.Phi[...,None]), function)
        print("Plotted results")


    print("PerCDL federated test completed.")



def test_our_model(X,signals,K,L,n_steps=200,save_path=None,percentile=None):

    print("Starting PerCDL test...")
    assert X.ndim == 2, "X should be 2D"

    fs = 95.95

    X = X[...,None]

    if percentile is not None:
       best_cutoff = find_optimal_cutoff(X, fs=fs, percentile=[percentile])
       X_filtered = apply_low_pass(X, best_cutoff, fs=fs,order=3)
       print(f"Applied low-pass filter with cutoff {best_cutoff} Hz")

    else:
        X_filtered = X


    X_filtered = torch.tensor(X_filtered, dtype=torch.float32)
    a,z,phi = tools.PerCDL(X,nb_atoms=K,atoms_length=L,n_iters=n_steps,n_perso=n_steps,signal_names=signals)
    print("Ran PerCDL model")
    print(f"Shapes - X: {X_filtered.shape}, z: {z.shape}, a: {a.shape}, phi: {phi.shape}")
    if save_path:
        plot_results(X_filtered, z, a, phi, None, save_path=save_path)
        print(f"Saved results to {save_path}")
    else:
        plot_results(X_filtered, z, a, phi, None)
        print("Plotted results")
    print("PerCDL test completed.")


@click.command()
@click.option('--atoms', default=2, help='Number of atoms K', type=int)
@click.option('--length', default=30, help='Atom length L', type=int)
@click.option('--steps', default=200, help='Number of steps for optimization', type=int)
@click.option('--step_size', default=1e-3, help='Step size for optimization', type=float)
@click.option('--percentile', default=None, help='Percentile for low-pass filter', type=float)
@click.option('--save_path', default=None, help='Path to save results', type=str)
@click.option('--channel', default=3, help='Number of signal channels', type=int)
def main(atoms, length, steps, step_size, percentile, save_path,channel):
    subjects = list(range(1, 11))  # 10 subjects
    signal_names = ['TOX','TAX','TAY','RAV','RAZ','RRY','LAV','LAZ','LRY']   # select only three signals
    signals = [signal_names[channel]]
    X = build_X(subjects, signals, trial=1)
    print("built X")
    X = X[:,:,0]
    print(f"X shape: {X.shape}")

    phi_init = np.random.randn(atoms, length)
    phi_init = phi_init / np.linalg.norm(phi_init, axis=1, keepdims=True)

    D = 3
    W = 10

    function = TransformationFunction(length,D,W)



    test_percdl_federated(X,function,phi_init=phi_init,K=atoms,L=length,n_steps=steps,step_size=step_size,save_path=save_path)
    test_our_model(X,signals,K=atoms,L=length,n_steps=steps,save_path=save_path,percentile=percentile)


if __name__ == "__main__":
    main()
    



    

    



    