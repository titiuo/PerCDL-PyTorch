
from jax import (
    jit, vmap, jacfwd, jvp
)
from functools import partial
from jax.lax import fori_loop
from jax import device_put
import jax.numpy as jnp


sigma = 0.02
constraint_limit = 0.99

#@partial(jit,static_argnums=[1,2,3])
def one_layer(params_i,arange_pi):
    m=len(params_i)
    def layer(psi):
        arg_sin = jnp.outer(psi,arange_pi)
        renorm_weights=jnp.reshape(params_i/arange_pi,(1,m))
        sin_params = (jnp.sin(arg_sin)*renorm_weights).sum(axis=1)
        return psi+sin_params
    return layer


#@partial(jit,static_argnums=[1,2,3])
def multiple_layer(a, D:int, W:int, L:int):
    
    params = jnp.reshape(a, (D, W))
    arange_pi = jnp.arange(W+1)[1:] * jnp.pi
    arange_time = jnp.arange(L) / (L-1)

    return fori_loop(0, D, lambda i, x: one_layer(params[i], arange_pi)(x), arange_time)

#@partial(jit,static_argnames=["L"])
def transform_x_by_psi(x,psi,L:int):
    """
    Take the common atom x and a time reparametrization,
    return x circ psi

    we assume that x(i/L)=x[i] and psi(i/L)=psi[i]
    
    """
    
    arange_time = jnp.arange(L)/(L-1)
    # sigma_squared = 10* 1/ L**2 #sigma
    sigma_squared = 0.0001
    print('sigma squared =', sigma_squared)

    psi_rep = jnp.repeat(psi.reshape(1,-1),L,axis=0).T
    
    weights = jnp.exp(-(psi_rep-jnp.reshape(arange_time,(1,-1)))**2/sigma_squared)# matrix of weight
    
    weights = weights/jnp.reshape(jnp.sum(weights,axis=1),(1,len(weights[0])))
    val = x[1:-1]+(psi[1:-1]-arange_time[1:-1])*(x[2:]-x[1:-1]) # we know that transfo (x,psi)[0]=x[0], (x,psi)[1]=x[1]
    x_transfo = jnp.concatenate([jnp.array([x[0]]),val,jnp.array([x[-1]])],axis=0)
    return jnp.matmul(weights, x_transfo)



def mat_psi(param_flatten,D:int,W:int,L:int):
    """

    Args:
        param_flatten (_type_): M parameter
        D (int): depth
        W (int): width
        L (int): common atom size

    Returns:
        LxL matrix related to the linear transformation induced by the personalisation
    """
    return vmap(lambda x_jnp: transform_x_from_params(x_jnp,param_flatten,D,W,L))(jnp.eye(L)).T




@jit
def projection_params(params_to_normalized):
    """
    take the parameter (numpy array) and renormalized it to ensure that psi is a diffeomorphism
    """
    
    # print("params_to_normalized =", params_to_normalized.shape)
    norm_1_params = jnp.linalg.norm(params_to_normalized, ord=1, axis=1)
    norm_1_params_= vmap(
        lambda x: jnp.where(x>constraint_limit, x, constraint_limit)
    )(norm_1_params).reshape(-1, 1)
    params_projected = constraint_limit*params_to_normalized/norm_1_params_

    return jnp.reshape(params_projected,(-1,)) # flatten shape

#@partial(jit,static_argnames=["nb_layers", "width", "L"])
def transform_x_from_params(x_jnp,param_flatten,nb_layers:int,width:int,L:int):
    """
    Args:
        x_jnp (_type_): atoms of size L
        param_flatten (_type_): parameter array of size M

    Returns:
        jnp.array: the reparametrized atom of size L
    """
    psi=multiple_layer(param_flatten,nb_layers,width,L)
        
    return transform_x_by_psi(x_jnp  , psi, L)

#@partial(jit,static_argnames=["D", "W", "L"])
def transform_x_from_all_params(phi, a, D:int, W:int, L:int):
    """
    Inputs:
        phi: common atom to personalize (L)
        a: personalisation parameters (S x M)
        nb_layers (int): nombre de couche pour le réseau de paramétrison
        width (int): largeur de couche pour le réseau de paramétrison
        L (int): taille du signal

    Returns:
        Array of personalized atoms (S x L)
    """
    Psi = vmap(lambda a_: multiple_layer(a_, D, W, L))(a)
    Phi_personalized = vmap(lambda psi: transform_x_by_psi(phi, psi, L))(Psi)
        
    return Phi_personalized

@partial(jit, static_argnames=["nb_layers", "width", "L"])
def _personalize(Phi, A, nb_layers:int, width:int, L:int):
    """
    Construct the personalized dictionary.

    Inputs:
        - Phi: K x L
        - A: K x S x M

    Output:
        Array of personalized atoms (K x S x L)
    """
    return vmap(
        lambda phi, a: transform_x_from_all_params(
            phi, a, nb_layers, width, L
        )
    )(Phi, A.transpose(1, 0, 2))

@partial(jit, static_argnames=["nb_layers", "width", "L"])
def _personalize_federated(Phi_ind, A_ind, nb_layers:int, width:int, L:int):
    """
    Construct the personalized dictionary.

    Inputs:
        - Phi: K x L
        - A: K x M

    Output:
        Array of personalized atoms (K  x L)
    """
    return vmap(
        lambda phi, a: transform_x_from_params(phi,a,nb_layers,width,L)
    )(Phi_ind, A_ind)

#@partial(jit,static_argnames=["nb_layers","width","L","m"])
def derive_partial_transform_x_from_params_m(x_jnp,param_flatten,nb_layers:int,width:int,L:int,m:int):
    to_derive=jit(lambda param_flatten:transform_x_from_params(x_jnp,param_flatten,nb_layers,width,L) )
    v=jnp.zeros(nb_layers*width)
    v_m=v.at[m].set(1)
    value_f,partial_derivative=jvp(to_derive,(param_flatten,),(v_m,))#take the function
    #partial derivative is J(f)_x v, the jacobienne time a vector J(f)_x\in \Rset^mxn if f: R^n -> R^m
    return partial_derivative




### compute jacobian
#@partial(jit,static_argnames=["nb_layers","width","L"])
def jac_D_base(atom,nb_layers:int,width:int,L:int):
    """"
    Take an atom
    and return the function alpha-> Jac_atom(f)(atom,alpha)
    
    """
    x_jnp=device_put(atom)
    def jac_to_atom_with_alpha(alpha):
        return jacfwd(lambda x : transform_x_from_params(x,alpha,nb_layers,width,L))(x_jnp)
    return jit(jac_to_atom_with_alpha)