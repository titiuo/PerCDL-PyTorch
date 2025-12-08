
from ..transformation_function.transformation import *
from jax.example_libraries import optimizers as jax_opt
from jax import value_and_grad, jit, vmap
from functools import partial
import jax.numpy as jnp
import optax
import jax.lax as lax
import numpy as np


def filter_Z(Z):
    # Z Sx K x N
    mask=((Z-np.max(Z,axis=1)[:,None,:])>=0)*1
    # for k in range(len(Z)):
    #     for s in range(len(mask)):
    #         nb_ac=np.sum(mask[s,k,:])
    #         if nb_ac==0:
    #             ind=np.argmax(Z[s,k])
    #             mask[s,k,ind]=0.01

    Z=Z*mask
    return Z

@partial(jit,static_argnames=["nb_layers", "width", "L"])
def recenter_Phi(Phi, A, nb_layers:int, width:int, L:int):

    Psi_mean = vmap(
        lambda x: Psi_k_mean(x, nb_layers, width, L)
    )(A.transpose(1, 0, 2)) 
    
    ### A.T because A (S,M,K) #A.transpose(2,0,1)=(K,S,M)
    PhiT = vmap(
        lambda x, psi: transform_x_by_psi(x, psi, L)
    )(Phi, Psi_mean)
    PhiT=PhiT/jnp.linalg.norm(PhiT,axis=1)[:,None]
    return PhiT


# I have replace Adam with adabelief a becktracking line search
@partial(jit,static_argnames=["nb_layers", "width", "L"])
def relearn_A(Phi_new, Phi_old, A, nb_layers:int, width:int, L:int):

    nb_steps = 40
    step_size = 0.01
    A_init = A

    D_personalised_old = vmap(
        lambda x, alpha: transform_x_from_all_params(x, alpha, nb_layers, width, L)
    )(Phi_old, A.transpose(1, 0, 2)) # We put K on the first axis
    # size (K, S, L)

    @jit
    def loss_to_opt(A_new):
        D_personalised_new = vmap(
            lambda x, alpha: transform_x_from_all_params(x, alpha, nb_layers, width, L)
        )(Phi_new, A_new.transpose(1, 0, 2))
        return jnp.linalg.norm(D_personalised_old-D_personalised_new)**2
    
    # simple gradient with line search
    solver = optax.chain(optax.adabelief(learning_rate=1.),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True
    ))
    opt_state = solver.init(A_init)
    value_and_grad = optax.value_and_grad_from_state(loss_to_opt)
    @jit
    def step(step, current):
        params,opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=loss_to_opt)
        params = optax.apply_updates(params, updates)
        
        params_to_normalized_K_S = jnp.reshape(params, (params.shape[0], -1, nb_layers, width))
        params = vmap(lambda x_S: vmap(projection_params)(x_S))(params_to_normalized_K_S) # Change here

        return (params,opt_state)

    
    # Optimize A
    A_final,opt_final = lax.fori_loop(0, nb_steps, step, (A_init,opt_state))

    return A_final


@partial(jit,static_argnames=["nb_layers", "width", "L"])
def Psi_k_mean(A_k, nb_layers:int, width:int, L:int):
    Psi_k_vec = vmap(
        lambda x: multiple_layer(x, nb_layers, width, L)
    )(A_k)
    Psi_k_mean_output = Psi_k_vec.mean(axis=0)
    return Psi_k_mean_output





