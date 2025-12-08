
from jax.example_libraries import optimizers as jax_opt
from jax import ( 
    jit, device_put, value_and_grad
)
import jax

from ..transformation_function import _personalize,mat_psi,_personalize_federated
from functools import partial
from .utils import l2_loss
import jax.numpy as jnp
import jax.lax as lax
import optax
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
from ott.geometry.costs import SoftDTW as soft
from dtaidistance import dtw_barycenter



def _CD_DTW(
    X, Phi, Z,
    step_size: float,
    nb_steps: int
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """
    
    Z_ = device_put(Z)
    #X_ = device_put(X)
    #Phi_init = device_put(Phi)
    L=len(Phi[0])
    K=len(Phi)
    cons=jnp.ones(L)
    convolve_result= vmap(
        lambda z: vmap(lambda h: jnp.convolve(h, cons, mode="full"))(z)
    )(Z_)
    arg=np.array(convolve_result)>0
    for k in range(K):
        
        obs_k=np.reshape(X[arg[:,k,:]],(-1,L))
        if  not obs_k.shape[0]==0:
            #Phi[k,:]=np.mean(obs_k,axis=0)
            #Phi[k,:]=np.array(soft(gamma=0.1).barycenter(weights=jnp.ones(len(obs_k))/len(obs_k),xs=jnp.array(obs_k)))
            
            Phi[k,:]=dtw_barycenter.dba(obs_k,c=Phi[k,:],use_c=False)
        

        # if k==0:
        #     plt.figure(k)
        #     plt.plot(obs_k[0])

        #     plt.plot(np.mean(obs_k,axis=0),label='mean')
        #     plt.plot(Phi[k],label="mean dtw")
            Phi[k,:]=Phi[k,:]/np.linalg.norm(Phi[k,:])
    #plt.legend()
    #plt.show()

    return Phi

#@partial(jit, static_argnames=["step_size", "nb_steps","dtw"])
def _CDU_easy(
    X, Phi, Z,
    step_size: float,
    nb_steps: int,dtw=False
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """
    
    Z_ = device_put(Z)
    #X_ = device_put(X)
    #Phi_init = device_put(Phi)
    L=len(Phi[0])
    K=len(Phi)
    cons=jnp.ones(L)
    convolve_result= vmap(
        lambda z: vmap(lambda h: jnp.convolve(h, cons, mode="full"))(z)
    )(Z_)
    arg=np.array(convolve_result)>0
    for k in range(K):
        
        obs_k=np.reshape(X[arg[:,k,:]],(-1,L))
        if  not obs_k.shape[0]==0:
            Phi[k,:]=np.mean(obs_k,axis=0)
            #Phi[k,:]=np.array(soft(gamma=0.1).barycenter(weights=jnp.ones(len(obs_k))/len(obs_k),xs=jnp.array(obs_k)))
            if dtw:
                Phi[k,:]=dtw_barycenter.dba(obs_k,c=Phi[k,:],use_c=False)
        

        # if k==0:
        #     plt.figure(k)
        #     plt.plot(obs_k[0])

        #     plt.plot(np.mean(obs_k,axis=0),label='mean')
        #     plt.plot(Phi[k],label="mean dtw")
            Phi[k,:]=Phi[k,:]/np.linalg.norm(Phi[k,:])
    #plt.legend()
    #plt.show()

    return Phi

#problème : peu robuste au bruit sur A
#utiliser une convolution pour smooth ? il faudrait faire une meilleur interpolation
# il y a également des effets de bords pas top

def filter_matrix(A):
    A_n=vmap(lambda l: jnp.where(l<jnp.max(l),0,1))(A.T).T
    return A_n
    
def _CDU_perso_easy2(
    X, Phi, Z,A,
    step_size: float,
    nb_steps: int,D:int,
    W:int,
    L:int,dtw=False
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """
    
    Z_ = device_put(Z)
    A_=device_put(A)
    #X_ = device_put(X)
    #Phi_init = device_put(Phi)
    L=len(Phi[0])
    K=len(Phi)
    cons=jnp.ones(L)
    convolve_result= vmap(
        lambda z: vmap(lambda h: jnp.convolve(h, cons, mode="full"))(z)
    )(Z_)
    arg=np.array(convolve_result)>0
    Matrix=vmap(lambda a_s : vmap(lambda a: filter_matrix(mat_psi(a,D,W,L)))(a_s))(A_)#SxKxLxL
    Matrix=np.array(Matrix)
    #print(Matrix[0,0])
    
    #print(np.mean(Z[0,0][Z[0,0]>0]))
    for k in range(K):
        Sigma=np.zeros((L,L))
        Y_Sigma=np.zeros(L)
        for s in range(len(Z)):
        
            acti=np.maximum(Z[s,k][Z[s,k]>0],np.ones(len(Z[s,k][Z[s,k]>0])))

            #acti=jnp.ones(len(acti))
            obs_s_k=np.reshape(X[s,arg[s,k,:]],(-1,L))/acti[:,None]
            p_s_k=len(obs_s_k)
            M_s_k=Matrix[s,k]
            Sigma=Sigma+p_s_k*np.matmul(M_s_k.T,M_s_k)
            Y_Sigma=Y_Sigma+np.sum(np.matmul(Matrix[s,k].T,obs_s_k.T),axis=1)

            
        #print(Sigma)
        #print(Sigma.shape,np.eye(L).shapen,Y_Sigma.shape)
        
        if np.linalg.cond(Sigma)>100:

            #Sigma=Sigma+np.eye(L)/len(Z)
            Sigma=0*9*Sigma+0.1*np.eye(L)#np.eye(L)/len(Z)

        Phi[k,:]=np.linalg.solve(Sigma,Y_Sigma)
        #Phi[k,:]=np.convolve(Phi[k,:], np.ones(3), 'same') / 3
        obs=np.reshape(X[arg[:,k,:]],(-1,L))
        if dtw:
            Phi[k,:]=dtw_barycenter.dba(obs,c=Phi[k,:],use_c=False)
        Phi[k,:]=Phi[k,:]/np.linalg.norm(Phi[k,:])

    #     if k==0:
    #         plt.figure(k)
    #         plt.plot(obs_k[0])
    #         plt.plot(Phi[k],label="mean")
    # plt.legend()
    # plt.show()

    return Phi

def _CDU_perso_IPU(
    X, Phi, Z,A,
    step_size: float,
    nb_steps: int,D:int,
    W:int,
    L:int,dtw=False):
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
    
    #possibly biased by the number of repitions per signal
    for s in range(S):
        for k in range(K):
            obs=np.reshape(X[s,arg[s,k,:]],(-1,L))
            if repet_numbers[s,k]==repet_max:
                Target[k,s]=obs
                mask[k,s]=np.ones(obs.shape)
            else: #step not useful but clear
                Target[k,s,:len(obs),:]=obs
                mask[k,s,:len(obs),:]=np.ones(obs.shape)

            acti=Z[s,k][Z[s,k]>0]
            if len(acti)>0:
                Amp[k,s]=np.mean(acti)
            else:
                Amp[k,s]=0
    mask=device_put(mask)
    Target=device_put(Target)
    Amp=device_put(Amp)
    solver_phi = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    solver_A = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    #solver=optax.lbfgs()
    opt_state_phi = solver_phi.init(Phi_)
    opt_state_A = solver_A.init(A_init)

    
    @jit
    def _loss(Phi_current,A_current):
        D_perso = _personalize(Phi_current, A_current, D, W, L)#KxSxL
        D_perso_amp_good=Amp[:,:,None]*D_perso
        return jnp.linalg.norm((Target-D_perso_amp_good[:,:,None,:])*mask) ** 2 
   
    
    # @jit
    # def _loss2(Phi_current):
    #     D_perso = _personalize(Phi_, A_current, D, W, L)
    #     return l2_loss(X_, Z_, D_perso) 
    # print(_loss(Phi_init),_loss2(A_init))

    @jit
    def _step(nb_step, current):
        params_phi,params_A,opt_state_phi,opt_state_A=current

        value_and_grad_phi = optax.value_and_grad_from_state(lambda ph:_loss(ph,params_A))
        value_and_grad_A = optax.value_and_grad_from_state(lambda pA:_loss(params_phi,pA))

        value_phi, grad_phi = value_and_grad_phi(params_phi, state=opt_state_phi)
        value_A, grad_A = value_and_grad_A(params_A, state=opt_state_A)
        #grad = jax.grad(_loss)(params)
        updates_phi, opt_state_phi = solver_phi.update(grad_phi, opt_state_phi, params_phi, value=value_phi, grad=grad_phi, value_fn=lambda ph:_loss(ph,params_A))
        updates_A, opt_state_A = solver_A.update(grad_A, opt_state_A, params_A, value=value_A, grad=grad_A, value_fn=lambda pA:_loss(params_phi,pA))
        #updates, opt_state = solver.update(grad, opt_state, params)
        params_phi = optax.apply_updates(params_phi, updates_phi)
        params_A = optax.apply_updates(params_A, updates_A)

       
    
        return (params_phi,params_A,opt_state_phi,opt_state_A)
        # Optimize A
    Phi_final,A_final,opt_phi_final,opt_A_final = lax.fori_loop(0, nb_steps, _step, (Phi_,A_init,opt_state_phi,opt_state_A))
    return Phi_final,A_final

def _CDU_perso_IPU_federated(
    X, Phi, Z, A,
    step_size: float,
    nb_steps: int,D:int,
    W:int,
    L:int,dtw=False):
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

    # print('repet_numbers =', repet_numbers)
    
    #possibly biased by the number of repitions per signal
    for s in range(S):
        for k in range(K):
            obs=np.reshape(X[s,arg[s,k,:]],(-1,L))
            if repet_numbers[s,k]==repet_max:
                Target[k,s]=obs
                mask[k,s]=np.ones(obs.shape)
            else: #step not useful but clear
                Target[k,s,:len(obs),:]=obs
                mask[k,s,:len(obs),:]=np.ones(obs.shape)

            acti=Z[s,k][Z[s,k]>0]
            if len(acti)>0:
                Amp[k,s]=np.mean(acti)
            else:
                Amp[k,s]=0
    mask=device_put(mask)
    Target=device_put(Target)
    Amp=device_put(Amp)
    solver_phi = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    solver_A = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    #solver=optax.lbfgs()
    #print(Phi_[None,:,:].repeat(S,axis=0).shape)
    #print(A_init.shape)
    Phi_ex=Phi_[None,:,:].repeat(S,axis=0)
    opt_state_phi = solver_phi.init(Phi_ex)
    opt_state_A = solver_A.init(A_init)
    
    
    @jit
    def _loss(Phi_current,A_current):
        D_perso = vmap(lambda Phi_ind,A_ind: _personalize_federated(Phi_ind, A_ind, D, W, L))(Phi_current,A_current)#SxKxL
        D_perso_amp_good=Amp[:,:,None]*D_perso.transpose(1,0,2)
        return jnp.linalg.norm((Target-D_perso_amp_good[:,:,None,:])*mask) ** 2 
   
    
    # @jit
    # def _loss2(Phi_current):
    #     D_perso = _personalize(Phi_, A_current, D, W, L)
    #     return l2_loss(X_, Z_, D_perso) 
    # print(_loss(Phi_init),_loss2(A_init))

    @jit
    def _step(nb_step, current):
        params_phi,params_A,opt_state_phi,opt_state_A=current

        value_and_grad_phi = optax.value_and_grad_from_state(lambda ph:_loss(ph,params_A))
        value_and_grad_A = optax.value_and_grad_from_state(lambda pA:_loss(params_phi,pA))

        value_phi, grad_phi = value_and_grad_phi(params_phi, state=opt_state_phi)
        value_A, grad_A = value_and_grad_A(params_A, state=opt_state_A)
        #grad = jax.grad(_loss)(params)
        updates_phi, opt_state_phi = solver_phi.update(grad_phi, opt_state_phi, params_phi, value=value_phi, grad=grad_phi, value_fn=lambda ph:_loss(ph,params_A))
        updates_A, opt_state_A = solver_A.update(grad_A, opt_state_A, params_A, value=value_A, grad=grad_A, value_fn=lambda pA:_loss(params_phi,pA))
        #updates, opt_state = solver.update(grad, opt_state, params)
        params_phi = optax.apply_updates(params_phi, updates_phi)
        params_A = optax.apply_updates(params_A, updates_A)
        return (params_phi,params_A,opt_state_phi,opt_state_A)

    # Optimize A
    Phi_final,A_final,opt_phi_final,opt_A_final = lax.fori_loop(0, nb_steps, _step, (Phi_ex,A_init,opt_state_phi,opt_state_A))
    
    repet_numbers_ = device_put(repet_numbers)
    # print('Phi before sum', Phi_final)
    # print('repet_numbers_mean', repet_numbers_mean)
    if np.sum(repet_numbers_) > 0:
        repet_numbers_mean=repet_numbers_/jnp.sum(repet_numbers_)#coefficient for aggregation
        Phi_final=jnp.sum(Phi_final*repet_numbers_mean[:,:,None],axis=0)#agregation step
    else:
        Phi_final = jnp.mean(Phi_final, axis=0)

    return Phi_final,A_final


def _CDU_perso_old_school(
    X, Phi, Z,A,
    step_size: float,
    nb_steps: int,D:int,
    W:int,
    L:int,dtw=False):
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
            acti=Z[s,k][Z[s,k]>0]
            if len(acti)>0:
                Amp[k,s]=np.mean(acti)
            else:
                Amp[k,s]=0
    mask=device_put(mask)
    Target=device_put(Target)
    Amp=device_put(Amp)
    solver = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    #solver=optax.lbfgs()
    opt_state = solver.init(Phi_)

    
    @jit
    def _loss(Phi_current):
        D_perso = _personalize(Phi_current, A_init, D, W, L)#KxSxL
        D_perso_amp_good=Amp[:,:,None]*D_perso
        return jnp.linalg.norm((Target-D_perso_amp_good[:,:,None,:])*mask) ** 2    
    value_and_grad = optax.value_and_grad_from_state(_loss)
    # @jit
    # def _loss2(Phi_current):
    #     D_perso = _personalize(Phi_, A_current, D, W, L)
    #     return l2_loss(X_, Z_, D_perso) 
    # print(_loss(Phi_init),_loss2(A_init))

    @jit
    def _step(nb_step, current):
        params,opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        #grad = jax.grad(_loss)(params)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
        #updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

       
    
        return (params,opt_state)
        # Optimize A
    Phi_final,opt_final = lax.fori_loop(0, nb_steps, _step, (Phi_,opt_state))
    return Phi_final
    


def _CDU_perso_easy(
    X, Phi, Z,A,
    step_size: float,
    nb_steps: int,D:int,
    W:int,
    L:int,dtw=False
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """
    
    Z_ = device_put(Z)
    A_=device_put(A)
    #X_ = device_put(X)
    #Phi_init = device_put(Phi)
    L=len(Phi[0])
    K=len(Phi)
    cons=jnp.ones(L)
    convolve_result= vmap(
        lambda z: vmap(lambda h: jnp.convolve(h, cons, mode="full"))(z)
    )(Z_)
    arg=np.array(convolve_result)>0
    Matrix=np.array(vmap(lambda a_s : vmap(lambda a: mat_psi(a,D,W,L))(a_s))(A_))#SxKxLxL
    
    #print(np.mean(Z[0,0][Z[0,0]>0]))
    for k in range(K):
        Sigma=np.zeros((L,L))
        Y_Sigma=np.zeros(L)
        for s in range(len(Z)):
        
            acti=np.minimum(Z[s,k][Z[s,k]>0],np.ones(len(Z[s,k][Z[s,k]>0])))

            #acti=jnp.ones(len(acti))
            obs_s_k=np.reshape(X[s,arg[s,k,:]],(-1,L))#/acti[:,None]
            p_s_k=len(obs_s_k)
            M_s_k=Matrix[s,k]
            Sigma=Sigma+p_s_k*np.matmul(M_s_k.T,M_s_k)
            Y_Sigma=Y_Sigma+np.sum(np.matmul(Matrix[s,k].T,obs_s_k.T),axis=1)

            
        #print(Sigma)
        #print(Sigma.shape,np.eye(L).shapen,Y_Sigma.shape)
        
        if np.linalg.cond(Sigma)>100:

            #Sigma=Sigma+np.eye(L)/len(Z)
            Sigma=0*9*Sigma+0.1*np.eye(L)#np.eye(L)/len(Z)

        Phi[k,:]=np.linalg.solve(Sigma,Y_Sigma)
        #Phi[k,:]=np.convolve(Phi[k,:], np.ones(3), 'same') / 3
        obs=np.reshape(X[arg[:,k,:]],(-1,L))
        if dtw:
            Phi[k,:]=dtw_barycenter.dba(obs,c=Phi[k,:],use_c=False)
        Phi[k,:]=Phi[k,:]/np.linalg.norm(Phi[k,:])

    #     if k==0:
    #         plt.figure(k)
    #         plt.plot(obs_k[0])
    #         plt.plot(Phi[k],label="mean")
    # plt.legend()
    # plt.show()

    return Phi

#I should code an
@partial(jit, static_argnames=["step_size", "nb_steps"])
def _CDU(
    X, Phi, Z,
    step_size: float,
    nb_steps: int
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """

    @jit
    def _loss(Phi_current):
        D = jnp.repeat(jnp.expand_dims(Phi_current, axis=1), X_.shape[0], axis=1)
        return l2_loss(X_, Z_, D)

    value_and_grad = optax.value_and_grad_from_state(_loss)
    @jit
    def _step(nb_step,current):
        params, opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
        params = optax.apply_updates(params, updates)
        return (params,opt_state)
    
    # JAX acceleration
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_init = device_put(Phi)

    # Initialize optimizer
    solver = optax.lbfgs()
    opt_state = solver.init(Phi_init)
    #opt_state = opt_init(Phi_init)

    # Optimize Phi 
    Phi_final,opt_final = lax.fori_loop(0, nb_steps, _step,(Phi_init, opt_state))
    
    # Return new Phi
    return Phi_final

#adabelief semble être plus rapide

@partial(jit, static_argnames=["step_size", "nb_steps", "D", "W", "L"])
def _CDU_perso(
    X, Phi, Z, A,
    step_size:float,
    nb_steps:int,
    D:int,
    W:int,
    L:int,dtw=False
):
    """
    Dictionary update step.

    Inputs:
        - Phi: K x L
        - A: S x K x M
        - X: S x N
        - Z: S x K x N
    """

    @jit
    def _loss(Phi_current):
        Phi_perso = _personalize(Phi_current, A_, D, W, L)
        return l2_loss(X_, Z_, Phi_perso)

    value_and_grad = optax.value_and_grad_from_state(_loss)
    @jit
    def _step(nb_step,current):
        params, opt_state=current
        value, grad = value_and_grad(params, state=opt_state)
        #grad = jax.grad(_loss)(params)
        updates, opt_state = solver.update(grad, opt_state, params, value=value, grad=grad, value_fn=_loss)
        #updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params,opt_state)
    
    # JAX acceleration
    A_ = device_put(A) 
    Z_ = device_put(Z)
    X_ = device_put(X)
    Phi_init = device_put(Phi)

    # Initialize optimizer
    #solver = optax.lbfgs()
    solver = optax.chain(optax.polyak_sgd(),optax.scale_by_backtracking_linesearch( max_backtracking_steps=15, store_grad=True))
    opt_state = solver.init(Phi_init)

    # Optimize Phi 
    Phi_final,opt_final = lax.fori_loop(0, nb_steps, _step,(Phi_init, opt_state))
    
    # Return new Phi
    return Phi_final