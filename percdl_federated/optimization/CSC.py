
import csc
from joblib import Parallel, delayed,parallel_backend
#should be run in parallel
def _CSC(X, Phi_perso, Z, S, penalty,backend="threading"):#backend = 'loky' if multiple processors

    #with parallel_backend('loky', inner_max_num_threads=1):
    #parallel = Parallel(n_jobs=-1, return_as="generator")
    function= lambda X,Phi: csc.update_z([X],
        dictionary=Phi, 
        penalty=penalty,
        constraint_str=csc.NO_CONSTRAINT
    )[0].T
    with parallel_backend('threading'):
        list_sol = Parallel(n_jobs=-1)(delayed(function)(X[s, :],Phi_perso.transpose(1, 0, 2)[s, :, :]) for s in range(S))
    
    for s in range(S):
        
        Z[s, :, :] = list_sol[s]
        

    return Z


# def _CSC(X, Phi_perso, Z, S, penalty,backend="threading"):#backend = 'loky' if multiple processors

#     #with parallel_backend('loky', inner_max_num_threads=1):
#     #parallel = Parallel(n_jobs=-1, return_as="generator")
#     function= lambda X,Phi: csc.update_z([X],
#         dictionary=Phi, 
#         penalty=penalty,
#         constraint_str=csc.NO_CONSTRAINT
#     )[0].T
#     # with parallel_backend('threading'):
#     #     list_sol = Parallel(n_jobs=-1)(delayed(function)(X[s, :],Phi_perso.transpose(1, 0, 2)[s, :, :]) for s in range(S))
    
#     for s in range(S):
        
#         Z[s, :, :] = function(X[s, :],Phi_perso.transpose(1, 0, 2)[s, :, :])
        

#     return Z