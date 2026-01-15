from ..optimization import (
    _CSC,_CDU_easy, _CDU, _CDU_perso, _IPU,pmap_IPU,_CDU_perso_easy,filter_Z,_IPU_easy,_IPU_reduce,_CD_DTW,
    normalize_Phi_Z, recenter_Phi, relearn_A,l2_loss,_CDU_perso_easy2,_CDU_perso_old_school,_CDU_perso_IPU,_CDU_perso_IPU_federated
)
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from ..transformation_function import _personalize, projection_params

import time 
class PerCDL:

    def __init__(
            self, f, X, K, L, 
            n_steps=100, step_size=1e-3, penalty=1e-2
        ) -> None:
        
        # Data parameters
        self.f = f
        self.X = X
        self.S = X.shape[0]
        self.N = X.shape[1]
        self.K = K
        self.L = L

        # Optimization parameters
        self.n_steps = n_steps
        self.step_size = step_size
        self.penalty = penalty

        # Initialize arrays of the correct shape with zeros
        self.reset()

    def reset(self):
        """
        Reset all arrays.
        """
        self.Phi_init = np.zeros((self.K, self.L))
        self.Z_init = np.zeros((self.S, self.K, self.N-self.L+1))
        self.A_init = np.zeros((self.S, self.K, self.f.M))
        self._initialized = False

    def initialize(self, Phi=None, Z=None, A=None):
        """
        Initialize arrays.
        """

        # Get initializations
        if Phi is not None:
            self.Phi_init = Phi
        if Z is not None:
            self.Z_init = Z
        if A is not None:
            self.A_init = A
        
        # Create copys
        self.Phi = self.Phi_init.copy()
        self.Z = self.Z_init.copy()
        self.A = self.A_init.copy()

        self._initialized = True

    def run(self):
        """
        Launch optimization process.
        """

        if not self._initialized:
            raise RuntimeError('PerCDL was not initialized.')

        @jit
        def _loss(X_, Z_,Phi_, A_current):
            D_perso = _personalize(Phi_, A_current, self.f.D, self.f.W, self.L)
            #assert X_.ndim == 2, "error X_ ndim"
            return l2_loss(X_, Z_, D_perso)   
        
        # Initial estimates
        print('Initial estimates:')
        #assert self.X.ndim == 2, "error X ndim"
        best_error=_loss(self.X, self.Z,self.Phi, self.A)
        count_increase=0
        count_decrease=0
        stop_bool=False
        for _ in range(self.n_steps):
            print(f'\tStep: {_+1}/{self.n_steps}')
            
            # Normalization
            if not(stop_bool):
                # print("step",_)
                self.Phi, self.Z = normalize_Phi_Z(self.Phi, self.Z)
                
                # CSC
                t1=time.time()
                Phi_perso = np.repeat(np.expand_dims(self.Phi, axis=1), self.S, axis=1)
                self.Z = self.CSC(Phi_perso)
                if _>0:
                    self.filter_Z()
                t2=time.time()
                print(f'\t\tCSC step: {t2-t1:.2f}s')
                # CDU (no personalization)
                t1=time.time()
                self.Phi = self.CDU()
                t2=time.time()
                print(f'\t\tCDU step: {t2-t1:.2f}s')
                new_error=_loss(self.X, self.Z,self.Phi, self.A)
                print(f'\t\tloss: {new_error:.2f}')

                if new_error>best_error:
                    count_increase=count_increase+1
                else:
                    count_decrease=count_decrease+1
                if new_error>best_error and count_increase==2:
                    count_decrease=0
                elif best_error>new_error and count_decrease==2:
                    count_increase=0
                if count_increase==3:
                    stop_bool=True

        # Personalization
        print('Personalization:')
        best_error=_loss(self.X, self.Z,self.Phi, self.A)
        count_increase=0
        count_decrease=0
        stop_bool=False
        
        #l'étape de barycenter peut avoir un effet régularisant trop fort
        #de même CDU old school peut être régularisant
        # for _ in range(5):
        for _ in range(self.n_steps):
            print(f'\tStep: {_+1}/{self.n_steps}')

            # Normalization
            if not(stop_bool):
                # CSC
                self.Phi, self.Z = normalize_Phi_Z(self.Phi, self.Z)
                t1=time.time()
                Phi_perso = self.f.personalize(self.Phi, self.A)
                self.Z = self.CSC(Phi_perso)
                t2=time.time()
                print(f'\t\tCSC step: {t2-t1:.2f}s')
                self.filter_Z()
                

                # print("erreur before CDU",_loss(self.X, self.Z,self.Phi, self.A))
                # Corriger cette étape
                # CDU (with personalization)
                # t1=time.time()
                # if _==self.n_steps-1:
                #     Phi = self.CDU_perso(dtw=True) # This Phi may not be centred, we'll correct this below
                # else:
                #     Phi = self.CDU_perso(dtw=False)
                # t2=time.time()
                # self.Phi=Phi

                if _==0:
                    t1=time.time()
                    A = self.IPU(self.Phi) # This A may not be centred either
                    t2=time.time()
                    print(f'\t\tIPU step: {t2-t1:.2f}s')
                    self.A=A
                    # print('A fter first IPU', self.A)

                t1=time.time()
                Phi,A=self.CDU_IPU(self.Phi)
                self.Phi=Phi
                self.A=A
                t2=time.time()
                print(f'\t\tCDU-IPU step: {t2-t1:.2f}s')
                # print('A =', self.A)
                # print('Phi =', self.Phi)
                # #self.Phi, self.Z = normalize_Phi_Z(Phi, self.Z)
                # print("CDUperso",t2-t1)
                # # IPU
                
                #print("erreur before IPU",_loss(self.X, self.Z,Phi, self.A))
                # t1=time.time()
                # A = self.IPU(Phi) # This A may not be centred either
                # t2=time.time()
                # print("IPU",t2-t1)
                # self.A=A

                # print("erreur before barycenter",_loss(self.X, self.Z,Phi, self.A))
                # Barycenter step
                
                t1=time.time()
                self.Phi = recenter_Phi(
                    Phi, A, self.f.D, self.f.W, self.L
                )
                #self.Phi=Phi
                self.A = relearn_A(
                    self.Phi, Phi, A, self.f.D, self.f.W, self.L
                )
                self.A=A
                t2=time.time()
                print(f'\t\tBarycenter step: {t2-t1:.2f}s')

                # CSC
                self.Phi, self.Z = normalize_Phi_Z(self.Phi, self.Z)
                t1=time.time()
                Phi_perso = self.f.personalize(self.Phi, self.A)
                self.Z = self.CSC(Phi_perso)
                t2=time.time()
                print(f'\t\tCSC step: {t2-t1:.2f}s')
                new_error=_loss(self.X, self.Z,self.Phi, self.A)
                if new_error>best_error:
                    count_increase=count_increase+1
                else:
                    count_decrease=count_decrease+1
                if new_error>best_error and count_increase==2:
                    count_decrease=0
                elif best_error>new_error and count_decrease==2:
                    count_increase=0
                if count_increase==3:
                    stop_bool=True
            new_error = _loss(self.X, self.Z,self.Phi, self.A)
            print(f'\t\tloss: {new_error:.2f}')


    def filter_Z(self):
        self.Z=filter_Z(self.Z)
    def normalize(self):
        """
        Normalization step.
        """
        return normalize_Phi_Z(self.Phi, self.Z)

    
    def CSC(self, Phi_perso):
        """
        One iteration of the CSC step.
        """
        return _CSC(self.X, Phi_perso, self.Z, S=self.S, penalty=self.penalty)
    def CD_DTW(self):
        return _CD_DTW(self.X, self.Phi, self.Z, step_size=self.step_size, nb_steps=25)


    def CDU(self,dtw=False):
        """
        One iteration of the CDU step.
        No personalization is considered.
        """
        return _CDU_easy(self.X, self.Phi, self.Z, step_size=self.step_size, nb_steps=25,dtw=dtw)
        #return _CDU_perso(self.X, self.Phi, self.Z, step_size=self.step_size, nb_steps=25)
    
    def CDU_perso(self,dtw=True):
        """
        One iteration of the CDU step.
        No personalization is considered.
        """
        return _CDU_perso_old_school(
            self.X, self.Phi, self.Z, self.A,
            step_size=self.step_size, nb_steps=100, 
            D=self.f.D, W=self.f.W, L=self.L,dtw=dtw
        )
        # return _CDU_perso_easy(self.X, self.Phi, self.Z, self.A,
        # step_size=self.step_size, nb_steps=100, 
        # D=self.f.D, W=self.f.W, L=self.L,dtw=dtw)
    #_CDU_perso_easy is not robust with bad values of A
    #CDU_perso_old_school and IPU_reduce bien ensemble mais bug pour S=500
    #idée, pourquoi ne pas apprendre A et Phi en même temps avec la méthode old school ?
    def IPU(self, Phi):
        """
        One iteration of the IPU step.
        """
        return _IPU_reduce(
            self.X, Phi, self.Z, self.A, 
            step_size=self.step_size, nb_steps=100,
            D=self.f.D, W=self.f.W, L=self.L
        )
    def CDU_IPU(self,Phi,dtw=False):
        return _CDU_perso_IPU_federated(self.X, Phi, self.Z, self.A,
            step_size=self.step_size, nb_steps=100, 
            D=self.f.D, W=self.f.W, L=self.L,dtw=dtw
        )
    
    def plot_common_atoms(self, toy_data):
        """
        Plot common atoms.
        """
        
        # Colors
        c_common_1 = 'cornflowerblue'
        c_common_2 = 'lightcoral'
        c_perso_1 = 'darkslateblue'
        c_perso_2 = 'crimson'

        _, ax = plt.subplots(figsize=(4, 4))

        true_patterns = [
            toy_data.first_pattern(),
            toy_data.second_pattern()
        ]


        for k, c_common, c_perso in zip(
            range(self.K), 
            [c_common_1, c_common_2],
            [c_perso_1, c_perso_2]
            ):
            
            # True common
            ax.plot(true_patterns[k] / np.linalg.norm(true_patterns[k]), 
                c=c_common, ls='--', lw=4, alpha=0.7, label=f'True common atom {k+1}'
            )

            # Common after optim
            ax.plot(self.Phi[k, :] / np.linalg.norm(self.Phi[k, :]), 
                c=c_perso, lw=1.5, alpha=0.8, label=f'Common atom {k+1}'
            )

        # Customization
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)   

        plt.tight_layout()
        plt.show()
    
    def plot_activations(self, toy_data):
        """
        Plot activations.
        """
        # Colors
        c_true_1 = 'mediumslateblue'
        c_true_2 = 'hotpink'
        c_perso_1 = 'darkslateblue'
        c_perso_2 = 'crimson'

        _, axes = plt.subplots(self.S, 1, figsize=(9, 3))

        for s, ax in enumerate(axes):

            # Input signal
            ax.axhline(c='lightgrey', ls='--', alpha=0.6)
            ax.plot(self.X[s, :], c='lightslategrey', lw=2, alpha=0.5, label='Input signal')

            for k, c_true, c_perso in zip(
                range(self.K), 
                [c_true_1, c_true_2],
                [c_perso_1, c_perso_2]
            ):
                once_true = True
                once_perso = True
                
                # True activations
                for j in np.arange(self.N-self.L):
                    z = toy_data.Z[s, k, j]
                    if z != 0:
                        ax.axvline(
                            j, alpha=0.5, lw=4, c=c_true,
                            label=f'True activations - Atom {k+1}' if (s==len(axes)-2 and once_true) else ''
                        )
                        once_true = False

                # Found activations
                alpha_max = abs(max(self.Z[s, k, :].min(), self.Z[s, k, :].max(), key=abs))
                for j in np.arange(self.N-self.L):
                    z = self.Z[s, k, j]
                    if z != 0:
                        alpha = abs(z / alpha_max)
                        if abs(z) >= 0.1:
                            ax.axvline(
                                j, alpha=alpha, lw=1.5, c=c_perso,
                                label=f'Activations - Atom {k+1}' if (s==len(axes)-2 and once_perso) else ''
                            )
                            once_perso = False

        # Customization
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_results(self, toy_data):
        """
        Plot optimization results (reconstruction + atoms).
        """

        # Define colors
        c_common_1 = 'cornflowerblue'
        c_common_2 = 'lightcoral'
        c_true_1 = 'mediumslateblue'
        c_true_2 = 'hotpink'
        c_perso_1 = 'darkslateblue'
        c_perso_2 = 'crimson'
        c_ax = 'lightslategrey'

        f = plt.figure(layout='constrained', figsize=(9, 3))
        gs = GridSpec(self.S, 5, figure=f)

        axes_signals = [
            f.add_subplot(gs[i, :-1])
            for i in range(self.S)
        ]
        axes_atoms = [
            f.add_subplot(gs[i, -1])
            for i in range(self.S)
        ]

        def _plot_reconstruction(axes):
            
            # Create time vector
            t = np.arange(self.X.shape[-1])

            # Get true atoms
            true_patterns = [
                toy_data.first_pattern(),
                toy_data.second_pattern()
            ]
            Phi_true = np.vstack(true_patterns)
            D_true = self.f.personalize(Phi_true, toy_data.A)

            # Plots for the inputs
            
            for s, ax in enumerate(axes):

                # Input signal
                ax.plot(t, self.X[s, :], c=c_ax, lw=4, alpha=0.5, label='Input signal')

                for k, c_true in zip(range(self.K), [c_true_1, c_true_2]):
                    
                    # Activations
                    for j in np.arange(self.N-self.L):
                        z = toy_data.Z[s, k, j]
                        if z != 0:
                            ax.axvline(j, alpha=0.5, lw=2, c=c_true)

                            # Reconstruction
                            ax.plot(t[j:j+self.L], D_true[k, s, :], c=c_true, lw=4, alpha=0.5)

            # Get personalized atoms
            D = self.f.personalize(self.Phi, self.A)

            # Plots
            for s, ax in enumerate(axes):

                for k, c_perso in zip(range(self.K), [c_perso_1, c_perso_2]):
                    once = True
                    
                    # Activations
                    alpha_max = abs(max(self.Z[s, k, :].min(), self.Z[s, k, :].max(), key=abs))
                    for j in np.arange(self.N-self.L):
                        z = self.Z[s, k, j]
                        if z != 0:
                            alpha = abs(z / alpha_max)
                            ax.axvline(j, alpha=alpha, lw=1.5, c=c_perso)

                            # Reconstruction
                            if abs(z) > 0.1:
                                ax.plot(t[j:j+self.L], z * D[k, s, :], c=c_perso, 
                                    label=f'Personalized atom {k}' if (s==len(axes)-1 and once) else ''
                                )
                                once = False

        def _plot_atoms(axes):

            # Get true atoms
            true_patterns = [
                toy_data.first_pattern(),
                toy_data.second_pattern()
            ]
            Phi_true = np.vstack(true_patterns)
            D_true = self.f.personalize(Phi_true, toy_data.A)

            # Plots for the input signal
            for i, ax in enumerate(axes):

                    # Horizontal line at 0
                    ax.axhline(c='lightgrey', alpha=0.5, ls='--')

                    for k, c_common, c_true in zip(
                        range(self.K), 
                        [c_common_1, c_common_2],
                        [c_true_1, c_true_2]
                    ):

                        # Common
                        ax.plot(
                            true_patterns[k] / np.linalg.norm(true_patterns[k]), 
                            c=c_common, ls='--', alpha=0.8, label=f'Common atom {k+1}'
                        )

                        # True
                        ax.plot(
                            D_true[k, i, :] / np.linalg.norm(D_true[k, i, :]), 
                            lw=4, c=c_true, alpha=0.5, label=f'True atom {k+1}'
                        )

            # Get personalized atoms
            D_perso = self.f.personalize(self.Phi, self.A)

            for i, ax in enumerate(axes):

                # Horizontal line at 0
                ax.axhline(c='lightgrey', alpha=0.5, ls='--')

                for k, c_perso in zip(range(self.K), [c_perso_1, c_perso_2]):
                    
                    # Perso
                    ax.plot(
                        D_perso[k, i, :] / np.linalg.norm(D_perso[k, i, :]), 
                        lw=1.3, c=c_perso, alpha=1, label=f'Perso. atom {k+1}'
                    )

            handles, labels = axes[1].get_legend_handles_labels()
            order = [0, 1, 4, 2, 3, 5]
            
            axes[1].legend(
                [handles[idx] for idx in order],[labels[idx] for idx in order],
                loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9
                )

        # Plot reconstruction
        _plot_reconstruction(axes_signals)

        # Plot atoms
        _plot_atoms(axes_atoms)

        # Customization
        for ax in (axes_signals + axes_atoms):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticklabels([])
        for ax in axes_signals[:-1]:
            ax.set_xticklabels([])
        for ax in axes_atoms[:-1]:
            ax.set_xticklabels([])

        plt.show()