# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook', 'grid'])

# Numba for speedup
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

#%% 50 x 50 lattice structure
N = 50

#%%
init_random = np.random.random((N,N))

# Define lattice_n with 75% spin-down and 25% spin-up particles
lattice_n = np.zeros((N, N))
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

# Define lattice_p with 75% spin-up and 25% spin-down particles
lattice_p = np.zeros((N, N))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1

#%% Plot the lattice in a contour map (raster)
plt.imshow(lattice_p)
plt.colorbar()

#%% Define dimensionless energy function E/J
def find_energy(lattice):
    # sums over first nearest-neighbours
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return arr.sum()

#%% Metropolis algorithm
@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_array, iters, BJ, energy):
    """
    Function to run the metropolis algorithm for arriving at the equilibrium
    state of the lattice.

    Args:
        spin_array - array defined as our lattice
        energy - initial energy of the 2D lattice
        iters - number of iterations for running the algorithm
        BJ - parameter defining the temperature of the heat reservoir
    """
    spin_array = spin_array.copy()
    net_spins = np.zeros(iters-1)
    net_energy = np.zeros(iters-1)
    for t in range(0,iters-1):
        # pick random point on lattice array and flip sign of spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_array[x,y] # initial spin
        spin_f = spin_i*-1 # proposed spin flip

        # find change in energy
        E_i = 0
        E_f = 0

        # Take care if the random point chosen happens to be around the
        # boundary of lattice
        if x>0:
            E_i += -spin_i*spin_array[x-1,y]
            E_f += -spin_f*spin_array[x-1,y]

        if x<N-1:
            E_i += -spin_i*spin_array[x+1,y]
            E_f += -spin_f*spin_array[x+1,y]

        if y>0:
            E_i += -spin_i*spin_array[x,y-1]
            E_f += -spin_f*spin_array[x,y-1]

        if y<N-1:
            E_i += -spin_i*spin_array[x,y+1]
            E_f += -spin_f*spin_array[x,y+1]

        # change state with designated probabilities
        dE = E_f - E_i
        if (dE>0)*(np.random.random() < np.exp(-BJ*dE)):
            spin_array[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y]=spin_f
            energy += dE

        net_spins[t] = spin_array.sum()
        net_energy[t] = energy

    return net_spins, net_energy

#%%
spins, energies = metropolis(lattice_p, 1000000, 0.3, find_energy(lattice_n))

#%% plotting
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins/N**2)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Average Spin $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energies)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.7', y=1.07, size=18)
plt.show()

#%%
def get_spin_energy(lattice, BJs):
    ms = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice, 1000000, bj, find_energy(lattice))
        ms[i] = spins[-100000:].mean()/N**2
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()
    return ms, E_means, E_stds

BJs = np.arange(0.1, 2, 0.05)
ms_n, E_means_n, E_stds_n = get_spin_energy(lattice_n, BJs)
ms_p, E_means_p, E_stds_p = get_spin_energy(lattice_p, BJs)

#%% Plot average spin m as a function of temperature T
plt.figure(figsize=(8,5))
plt.plot(1/BJs, ms_n, 'o--', label='75% of spins started negative')
plt.plot(1/BJs, ms_p, 'o--', label='75% of spins started positive')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$')
plt.ylabel(r'$\bar{m}$')
plt.legend(facecolor='white', framealpha=1)
plt.show()
