
from jax.example_libraries import optimizers as jax_opt
from jax import ( 
    jit, device_put, value_and_grad, vmap,pmap
)
from ..transformation_function import _personalize, projection_params,transform_x_from_params
from functools import partial
from .utils import l2_loss,l2_loss_ind
import jax.numpy as jnp
import jax.lax as lax
import optax
import jax
from joblib import Parallel, delayed,parallel_backend


@partial(jit, static_argnames=["step_size", "nb_steps", "D", "W", "L"])
def _IPU(
    X, Phi, Z, A,
    step_size:float, nb_steps:int,
    D:int, W:int, L:int
):
    """
    Parameters update step.

    Inputs:
        - X: S x N
        - Phi: K x L
        - Z: S x K x N
        - A: S x K x M
    """
    
    # @jit
    # def _loss(A_current):
    #     D_perso = _personalize(Phi_, A_current, D, W, L)
    #     return l2_loss(X_, Z_, D_perso)    
    # value_and_grad = optax.value_and_grad_from_state(_loss)
    # @jit
    # def _step(nb_step, current):
    #     params,opt_state=current
    #     value, grad = value_and_grad(params, state=opt_state)
    #     #grad = jax.grad(_loss)(params)
    #     updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
    #     #updates, opt_state = solver.update(grad, opt_state, params)
    #     params = optax.apply_updates(params, updates)

    #     # Projection step
    #     params = vmap(lambda x_S: vmap(lambda x: proj(x, D, W))(x_S))(params)
    
    #     return (params,opt_state)
    
    # JAX acceleration
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_ = device_put(Phi)
    A_init = device_put(A)
    

    
    # Initialize optimizer
    #solver = optax.lbfgs()# maybe problem since we project
    #solver=optax.adabelief(step_size)
    #solver = optax.chain(optax.adabelief(learning_rate=1.),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    solver = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    opt_state = solver.init(A_init)
    @jit
    def _loss(A_current):
        D_perso = _personalize(Phi_, A_current, D, W, L)
        return l2_loss(X_, Z_, D_perso)    
    value_and_grad = optax.value_and_grad_from_state(_loss)
    @jit
    def _step(nb_step, current):
        params,opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        #grad = jax.grad(_loss)(params)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
        #updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Projection step
        params = vmap(lambda x_S: vmap(lambda x: proj(x, D, W))(x_S))(params)
    
        return (params,opt_state)
        # Optimize A
    A_final,opt_final = lax.fori_loop(0, nb_steps, _step, (A_init,opt_state))
        
    
    
    
    # Return new A
    return A_final

@partial(jit, static_argnames=["D", "W"])
def proj(params, D:int, W:int):
    return projection_params(jnp.reshape(params, (D, W)))


@partial(jit, static_argnames=["step_size", "nb_steps", "D", "W", "L"])
def pmap_IPU(
    X, Phi, Z, A,
    step_size:float, nb_steps:int,
    D:int, W:int, L:int
):
    """
    Parameters update step.

    Inputs:
        - X: S x N
        - Phi: K x L
        - Z: S x K x N
        - A: S x K x M
    """
    
    # @jit
    # def _loss(A_current):
    #     D_perso = _personalize(Phi_, A_current, D, W, L)
    #     return l2_loss(X_, Z_, D_perso)    
    # value_and_grad = optax.value_and_grad_from_state(_loss)
    # @jit
    # def _step(nb_step, current):
    #     params,opt_state=current
    #     value, grad = value_and_grad(params, state=opt_state)
    #     #grad = jax.grad(_loss)(params)
    #     updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
    #     #updates, opt_state = solver.update(grad, opt_state, params)
    #     params = optax.apply_updates(params, updates)

    #     # Projection step
    #     params = vmap(lambda x_S: vmap(lambda x: proj(x, D, W))(x_S))(params)
    
    #     return (params,opt_state)
    
    # JAX acceleration
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_ = device_put(Phi)
    A_init = device_put(A)
    

    def optim_ind(X_s,Z_s,A_s):

        # Initialize optimizer
        solver = optax.lbfgs()# maybe problem since we project
        #solver=optax.adabelief(step_size)
        opt_state = solver.init(A_s)
        @jit
        def _loss(A_current_s):
            D_perso_s = vmap(lambda phi,a:transform_x_from_params(phi,a,D,W,L))(Phi_,A_current_s)
            return l2_loss_ind(X_s, Z_s, D_perso_s)    
        value_and_grad = optax.value_and_grad_from_state(_loss)
    
        
        @jit
        def _step(nb_step, current):
            params,opt_state=current
            value, grad = value_and_grad(params, state=opt_state)
            #grad = jax.grad(_loss)(params)
            updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
            #updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Projection step
            params = vmap(lambda x: proj(x, D, W))(params)
        
            return (params,opt_state)
        # Optimize A
        A_final_s,opt_final = lax.fori_loop(0, nb_steps, _step, (A_s,opt_state))
        return A_final_s

    # with parallel_backend('threading'):
    #     list_sol=Parallel(n_jobs=-1)(delayed(optim_ind)(X_[s],Z_[s],A_init[s]) for s in range(len(X_)))

    
    # for s in range(len(X_)):
    #     A[s]=list_sol[s]
    # A_final=A
    A_final=pmap(optim_ind)(X_,Z_,A_init)
    
    # Return new A
    return A_final

from dtaidistance import dtw_barycenter
import numpy as np
import matplotlib.pyplot as plt

#@partial(jit, static_argnames=["step_size", "nb_steps", "D", "W", "L"])
def _IPU_reduce(
    X, Phi, Z, A,
    step_size:float, nb_steps:int,
    D:int, W:int, L:int):
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_ = device_put(Phi)
    A_init = device_put(A)
    L=len(Phi[0])
    K=len(Phi)
    S=len(A_init)
    cons=jnp.ones(L)
    convolve_result= vmap(
        lambda z: vmap(lambda h: jnp.convolve(h, cons, mode="full"))(z)
    )(Z_)
    arg=convolve_result>0
    
    
    repet_numbers=np.zeros((S,K))
    for s in range(S):
        for k in range(K):
            repet=len(np.reshape(X[s,arg[s,k,:]],(-1,L)))
            repet_numbers[s,k]=repet
    repet_max=int(repet_numbers.max())
    #assume same number of repetitions per signal
    Target=np.zeros((K,S,repet_max,L))
    mask=np.zeros((K,S,repet_max,L))
    Amp=np.zeros((K,S))
    
    for s in range(S):
        for k in range(K):
            obs=np.reshape(X[s,arg[s,k,:]],(-1,L))
            if repet_numbers[s,k]==repet_max:
                Target[k,s]=obs
                mask[k,s]=np.ones(obs.shape)
            else: #step not useful but clear
                Target[k,s,:len(obs),:]=obs
                mask[k,s,:len(obs),:]=np.ones(obs.shape)
            if len(Z[s,k][Z[s,k]>0])>0:
                Amp[k,s]=np.mean(Z[s,k][Z[s,k]>0])
            else:
                Amp[k,s]=0
    Target=device_put(Target)
    Amp=device_put(Amp)
    mask=device_put(mask)
    solver = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    opt_state = solver.init(A_init)

    
    @jit
    def _loss(A_current):
        D_perso = _personalize(Phi_, A_current, D, W, L)#KxSxL
        D_perso_amp_good=Amp[:,:,None]*D_perso
        return jnp.linalg.norm((Target-D_perso_amp_good[:,:,None,:])*mask) ** 2    
    value_and_grad = optax.value_and_grad_from_state(_loss)
    @jit
    def _loss2(A_current):
        D_perso = _personalize(Phi_, A_current, D, W, L)
        return l2_loss(X_, Z_, D_perso) 
    print(_loss(A_init),_loss2(A_init))

    @jit
    def _step(nb_step, current):
        params,opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        #grad = jax.grad(_loss)(params)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
        #updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Projection step
        params = vmap(lambda x_S: vmap(lambda x: proj(x, D, W))(x_S))(params)
    
        return (params,opt_state)
        # Optimize A
    A_final,opt_final = lax.fori_loop(0, nb_steps, _step, (A_init,opt_state))
    return A_final


# Is slow because dtw barycenter is slow
def _IPU_easy(
    X, Phi, Z, A,
    step_size:float, nb_steps:int,
    D:int, W:int, L:int):
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_ = device_put(Phi)
    A_init = device_put(A)
    L=len(Phi[0])
    K=len(Phi)
    S=len(A_init)
    cons=jnp.ones(L)
    convolve_result= vmap(
        lambda z: vmap(lambda h: jnp.convolve(h, cons, mode="full"))(z)
    )(Z_)
    arg=convolve_result>0
    Phi_perso = _personalize(Phi_, A_init,D,W,L)
    Target=np.zeros((S,K,L))
    Amp=np.zeros((K,S))

    def func(obs_s_k,PHII):
        if len(obs_s_k)>0:
            #return dtw_barycenter.dba(obs_s_k,c=PHII,use_c=False)
            
            return np.mean(obs_s_k,axis=0)
        else: 
            return PHII


    # with parallel_backend('threading'):
    #     list_sol=Parallel(n_jobs=-1)(delayed(func)(np.reshape(X[s,arg[s,k,:]],(-1,L)),Phi_perso[s,k,:]) for s in range(S) for k in range(K))
    Target=np.zeros((K,S,L))
    for s in range(S):
        for k in range(K):
            acti=Z[s,k][Z[s,k]>0]
            if len(acti)>0:
                Amp[k,s]=np.mean(acti)
            else:
                Amp[k,s]=0
            Target[k,s]=func(np.reshape(X[s,arg[s,k,:]],(-1,L))/acti[:,None],Phi_perso[s,k,:])
            
    Amp=device_put(Amp)
            # plt.figure(s)
            # plt.plot(Target[k,s])
            # plt.show()
    #Target=np.reshape(np.array(list_sol),(K,S,L)) #we construct some target before to lean the parameter, however this process seems not easy
    # for s in range(S):
    #     for k in range(K):
    #         Target[s,k]=list_sol[s,k]

    Target=jnp.array(Target)
    solver = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    opt_state = solver.init(A_init)

    @jit
    def _loss(A_current):
        D_perso = _personalize(Phi_, A_current, D, W, L)
        D_perso_amp=D_perso#*Amp[:,:,None]
        return jnp.linalg.norm(Target-D_perso_amp)**2    
    value_and_grad = optax.value_and_grad_from_state(_loss)

    @jit
    def _step(nb_step, current):
        params,opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        #grad = jax.grad(_loss)(params)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
        #updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Projection step
        params = vmap(lambda x_S: vmap(lambda x: proj(x, D, W))(x_S))(params)
    
        return (params,opt_state)
        # Optimize A
    A_final,opt_final = lax.fori_loop(0, nb_steps, _step, (A_init,opt_state))
    return A_final
    


