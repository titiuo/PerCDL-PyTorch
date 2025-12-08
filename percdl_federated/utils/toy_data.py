
import matplotlib.pyplot as plt
import numpy as np
import itertools
from jax import vmap
import jax.numpy as jnp


class ToyData:

    def __init__(self, f, N, S, K, L, n,seed=0) -> None:
        """
        Helper class to generate a synthetic dataset.
        
        Inputs:
            - f: transformation function
            - N: length of each signals 
            - S: number of signals
            - K: number of atoms
            - L: Atom length
            - n: number of repetitions for each atom
        """   

        # Dataset parameters
        self.N = N
        self.S = S
        self.K = K
        self.L = L
        self.n = n
        self.seed=seed

        # Transformation function
        self.f = f

        # Initialize output arrays
        self.X = np.zeros((S, N))
        self.A = np.zeros((S, K, f.M))
        self.Z = np.zeros((S, K, N-L+1))

    def first_pattern(self):
        return np.sin(2*2*np.pi/(self.L-1)*np.arange(self.L))
    def second_pattern(self):
        half = self.L // 2
        arc1 = np.abs(np.sin(np.pi/(half)*np.arange(half)))
        arc2 = 0.3 * np.abs(np.sin(np.pi/(half-1)*np.arange(self.L-half)))
        return np.ravel([arc1, arc2])
    
    def generate(self):
        """
        Generate a synthetic dataset given the input parameters.

        Outputs:
            - X: signal (S x N)
            - Z: activations (S x K x N) 
            - A: parameters matrix (S x K x M)
        """
        
        # Generate random activations
        self._generate_activations2()

        # Initialize random number generator
        rng = np.random.default_rng(seed=self.seed)
        #à améliorer
        # Plant the patterns in each signals
        Phi_perso=np.zeros((self.S,self.K,self.L))
        for s in range(self.S):

            # Loop over atoms in each matrix
            for k in range(self.K):

                # Create random parameters
                alphas = 2*rng.random(self.f.M) -1
                alphas = self.f.proj(alphas)
                if k == 0:
                    x = self.first_pattern()
                if k == 1:
                    x = self.second_pattern()
                        # Transform base shape
                Phi_perso[s, k] = self.f.transform(x, alphas)
                self.A[s, k, :] = alphas

        convolve_result= vmap(lambda z, d: vmap(lambda h, j: jnp.convolve(h, j, mode="full"))(z, d))(self.Z, Phi_perso)
        self.X=np.array(convolve_result.sum(axis=1))
                # print('alphas =', alphas)

                # Loop o

                # Store the parameter in A
        self.A[s, k, :] = alphas

        return self.X, self.Z, self.A

    def _generate_activations2(self):
        rng = np.random.default_rng(seed=self.seed)
        place=self.N-self.K*self.n*self.L-self.L
        
        inter_place=place//(self.K*self.n)
        inter_place_random=rng.integers(inter_place,size=(self.S,self.K*self.n))
        
        
        for s in range(self.S):
            order=rng.choice(np.concatenate([np.zeros(self.n)+i for i in range(self.K)]),replace=False,size=(self.K*self.n))
            
            sum=0
            #print(self.Z.shape)
            for o in range(self.K*self.n):
                sum=sum+inter_place_random[s,o]
                
                self.Z[s,int(order[o]),sum]=1
                sum=sum+self.L
        
            

    def _generate_activations(self):
        """
        Generate atom activations.

        Output:
            - Z: activations (S x K x N) 
        """
        
        # Initialize random number generator
        rng = np.random.default_rng(seed=self.seed)
        max_iter = 500
        
        # Process each time series independently
        for s in range(self.S):

            # Track atom position
            choices = np.arange(self.N-self.L+1)
            forbidden = []
            
            # Process each atom
            for k in range(self.K):
                
                # Prune all indices in the range [N-L, N]
                choices = choices[choices <= self.N-self.L]

                # The atom cannot be within L of a forbidden interval
                for interval in forbidden:
                    interval[0] -= self.L
                    mask = (choices <= interval[0]) | (choices >= interval[1])
                    choices = choices[mask]

                # Get assignment. Note that the random numbers should
                # not be closer than the pattern length
                for i_try in range(max_iter):
                    assignment = rng.choice(choices, size=self.n)
                    
                    # Sort assignment
                    assignment = np.sort(assignment)
        
                    # Check distance between successive start values
                    distance = np.abs(np.diff(assignment))
                    if (distance >= self.L).all():
                        break
        
                # Fail condition
                if i_try == max_iter - 1:
                    raise RuntimeError(
                        "Couldn't assign activations. Try reducing the number of atoms and/or their lengths."
                    )
                
                # Create forbidden intervals
                forbidden_intervals = [
                    (s, s+self.L) for s in assignment
                ]
                # Also add the forbidden intervals found during the
                # previous step
                forbidden_intervals = forbidden_intervals + list(forbidden)
        
                # Sort intervals
                forbidden_intervals = np.sort(forbidden_intervals)
        
                # Retrieve allowed intervals
                allowed_intervals = [
                    list(range(i[1], j[0]))
                    for i, j
                    in zip(forbidden_intervals[:-1], forbidden_intervals[1:])
                ]
                # Let's not forget to add the first and last intervals
                allowed_intervals = list(range(0, forbidden_intervals[0][0])) +\
                    list(itertools.chain.from_iterable(allowed_intervals)) +\
                    list(range(forbidden_intervals[-1][1], self.N))
                # print("allowed_intervals =", allowed_intervals)
        
                # Update possible atom starts by merging the intervals
                choices = np.fromiter(
                    itertools.chain(allowed_intervals), dtype=int
                )       

                # Update activations
                self.Z[s, k, assignment] = 1

                # Update forbidden intervals
                forbidden = forbidden_intervals

    def plot_patterns(self):
        """
        Plot true patterns and their transformation.
        """

        _, axes = plt.subplots(self.S, 1, figsize=(5, 2*self.S))

        # True patterns
        pattern_1 = self.first_pattern()
        pattern_2 = self.second_pattern()

        # Transformation of the true patterns
        Phi_perso = self.f.personalize(np.vstack([pattern_1, pattern_2]), self.A)

        for s, ax in enumerate(axes):
            ax.plot(
                pattern_1 / np.linalg.norm(pattern_1), 
                label='Pattern 1', c='mediumslateblue', ls='--'
            )
            ax.plot(
                pattern_2 / np.linalg.norm(pattern_2), 
                label='Pattern 2', c='crimson', ls='--'
            )

            ax.plot(
                Phi_perso[0, s, :] / np.linalg.norm(Phi_perso[0, s, :]), 
                label='Transformed pattern 1', c='mediumslateblue'
            )
            ax.plot(
                Phi_perso[1, s, :] / np.linalg.norm(Phi_perso[1, s, :]),  
                label='Transformed pattern 2', c='crimson'
            )

            ax.axhline(c='lightslategrey', ls='--', alpha=0.3)
            ax.axvline(self.L//2, c='lightslategrey', ls='--', alpha=0.3)

        axes[self.S//2].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def plot_dataset(self):
        """
        Plot dataset.
        """

        _, axes = plt.subplots(self.S, 1, figsize=(8, 2*self.S))

        # True patterns
        patterns = [
            self.first_pattern(),
            self.second_pattern()
        ]

        # Transformation of the true patterns
        Phi_perso = self.f.personalize(np.vstack(patterns), self.A)

        t = np.arange(self.N)
        for s, ax in enumerate(axes):

            # Input signal
            ax.plot(t, self.X[s, :], c='darkslategrey', lw=2, alpha=0.5, label='Input signal')

            for k in range(self.K):
                
                # Iterate over the activations values
                for j in np.arange(self.N-self.L):
                    z = self.Z[s, k, j]
                    if z != 0:
                        # Plot activations
                        ax.axvline(
                            j, alpha=0.5, lw=2, c='mediumslateblue' if k==0 else 'crimson'
                        )
                        # Plot atom
                        ax.plot(
                            t[j:j+self.L], Phi_perso[k, s, :],
                            c='mediumslateblue' if k==0 else 'crimson', lw=2, alpha=0.5
                        )

        plt.show()