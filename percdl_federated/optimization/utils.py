
import jax.numpy as jnp
from jax import (
    jit, vmap
)
import numpy as np

# we can normalize according to the personalization
def normalize_Phi_Z(Phi, Z):
    """
    Normalize each atom in Phi, and scale Z accordingly.

    Inputs:
        Phi: K x L
        Z: S x K x L
    """

    # Get the norms
    norms = np.linalg.norm(Phi, axis=1)

    # Normalize
    Phi = np.divide(Phi, np.expand_dims(norms, axis=1))

    # Scale Z
    Z = np.multiply(Z, np.expand_dims(norms, axis=(0, 2)))

    return Phi, Z

#@jit
def reconstruct(Z, D):
    """
    Reconstruct the input signal from the activations and the
    personalized dictionary.

    Inputs:
        - Z: S x K x (N-L+1)
        - D_perso: K x S x L
    """

    D = D.transpose(1, 0, 2)
    convolve_result= vmap(
        lambda z, d: vmap(lambda h, j: jnp.convolve(h, j, mode="full"))(z, d)
    )(Z, D)

    return convolve_result.sum(axis=1)

def reconstruct_ind(Z_s, D_s):
    """
    Reconstruct the input signal from the activations and the
    personalized dictionary.

    Inputs:
        - Z: K x (N-L+1)
        - D_perso: K  x L
    """

    #D = D_.transpose(1, 0, 2)
    convolve_result= vmap(lambda h, j: jnp.convolve(h, j, mode="full"))(Z_s, D_s)

    return convolve_result.sum(axis=0)







#@jit
def l2_loss(X, Z, D):
    """
    l2 loss.

    Inputs:
        - X: S x N
        - Z: S x K x (N-L+1)
        - D: K x S x L
    """

    X_recon = reconstruct(Z, D)   
    
    return jnp.linalg.norm(X - X_recon) ** 2


def l2_loss_ind(X_s, Z_s, D_s):
    """
    l2 loss.

    Inputs:
        - X: S x N
        - Z: S x K x (N-L+1)
        - D: K x S x L
    """

    X_recon = reconstruct_ind(Z_s, D_s)
    
    return jnp.linalg.norm(X_s - X_recon) ** 2
