# optimised monte carlo by introducing classes and energy functions...
import numpy as np
import matplotlib.pyplot as plt

# define an ising model class to simplify calculations...
class ising_model_2D:
    # inputs our lattice, keeps the data of our lattice's energy and spin tagged
    def __init__(self, d, a, mfield):
        # lattice with a:(1-a) down-up
        temp = np.random.random((d, d))

        temp[temp >= a] = 1
        temp[temp < a] = -1

        # create 0 in the border (just to make neighbour atom calculations easier)
        temp = np.pad(array=temp, pad_width=1, mode="constant")
        self.lattice = temp.copy()
        del temp

        self.b = mfield

        # initialise the energy n net spin of our lattice
        self.energy = self.get_energy()  # use it to get monte carlo data.
        self.spin = np.sum(self.lattice)

    # to find the stable energy of our lattice
    def stable_energy(self):
        # the most stable energy(alligned spins))
        n = np.ones((d, d))
        n = np.pad(array=n, pad_width=1, mode="constant")

        e = 0
        # look at the i,jth atom.
        for i in range(1, d + 1):
            for j in range(1, d + 1):
                # see the neighbouring atoms(up,down,left,right)
                g = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

                # calculate the sigma_ij*sigma_i'j' around our atom at ij
                m = -n[i, j] * g * n[i - 1 : i + 2, j - 1 : j + 2]

                # sum over all the neighbour sigmas
                e += np.sum(m)

        return e

    # fn to find the energy of our lattice with B
    def get_energy(self):
        e = 0
        # look at the i,jth atom.
        for i in range(1, d + 1):  # correction: d->d+1..!
            for j in range(1, d + 1):
                # see the neighbouring atoms(up,down,left,right)
                g = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

                # calculate the sigma_ij*sigma_i'j' around our atom at ij
                m = -self.lattice[i, j] * g * self.lattice[i - 1 : i + 2, j - 1 : j + 2]

                # sum over all the neighbour sigmas
                e += np.sum(m) - self.b * self.lattice[i, j]

        return e

    # fn to calculate the energy of lattice on flipping spin at (x,y)
    def flipcheck(self, x, y):
        # the energy change comes only from the neighbours of the spin we flipped at (x,y)
        g = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # calculate the sigma_ij*sigma_i'j' around our atom at ij
        m = -self.lattice[x, y] * g * self.lattice[x - 1 : x + 2, y - 1 : y + 2]

        # sum over all the neighbour sigmas and the effect of B
        e1 = np.sum(m) - self.lattice[x, y] * self.b

        # flip the (x,y)th spin
        self.lattice[x, y] *= -1

        # print('in flip fn', self.lattice)

        # calculate the sigma_ij*sigma_i'j' around our atom at ij
        m = -self.lattice[x, y] * g * self.lattice[x - 1 : x + 2, y - 1 : y + 2]

        # sum over all the neighbour sigmas and the effect of B
        e2 = np.sum(m) - self.lattice[x, y] * self.b

        # flip bacc the (x,y)th spin
        self.lattice[x, y] *= -1
        # print('after flip fn', self.lattice)
        return e1, e2

    # fn to flip the (x,y)th spin
    def flip(self, x, y):
        # spin at (x,y) flippd
        self.lattice[x, y] *= -1

        # update the energy of our system after the iterations...
        e1, e2 = self.flipcheck(x, y)

        # the energy difference is double counted for all the neighbouring bonds
        self.energy += 2 * (e1 - e2)

        # update spin
        self.spin += 2 * self.lattice[x, y]


# dimension of our array
d = 200
# optional addition of external magnetic field B
b = 0.0

lat = ising_model_2D(d, 0.5, b)

print(lat.get_energy(), lat.spin, "starting!")
lattice = lat.lattice.copy()
plt.imshow(lattice, cmap="Greys_r", interpolation="nearest", origin="lower")

# metropolis algorithm
def metro(lat, reps):
    # our inverse temperature
    beta = 0.75

    # to record our monte carlo data
    iterate, energy, spin = np.zeros(reps), np.zeros(reps), np.zeros(reps)

    for i in range(reps):

        # random pick of our point in lattice
        x, y = np.random.randint(1, d), np.random.randint(1, d)

        # energies of our two systems
        e1, e2 = lat.flipcheck(x, y)

        # if energy increase only keep with a probability(more energy increase, lesser the prob)
        if e2 > e1:
            # probability of flipping if unfavourable
            p = np.exp(-beta * (e2 - e1))

            if np.random.random() < p:

                lat.flip(x, y)

        # flip the lattice spin if energy decreases
        else:
            lat.flip(x, y)

        # save the energy in case yu need to see the convergence of monte carlo..
        energy[i] = lat.energy
        iterate[i] = i
        spin[i] = lat.spin

    return iterate, energy, spin


x, y, z = metro(lat, 1000000)

# if you wanna plot the energy plots for every monte iteration, use this
plt.figure()
plt.grid()
plt.plot(x, y, label="Energy vs # iterations")
plt.legend()
plt.show()

plt.figure()
plt.grid()
plt.plot(x, z, label="Net Spin vs # iteration")
plt.legend()
plt.show()

print(lat.energy, lat.spin)
plt.imshow(lat.lattice, cmap="Greys_r", interpolation="nearest", origin="lower")

# this is one such generation of our ising model once it attained sufficient iterations.
# generate as much data as you require for all sets of temperatures.(by varying beta in metro algo...)

# try out animating the plots as monte carlo runs:
# at the end of each iteration of metro algo, we have a new state of our spin system as a numpy matrix of 1 and -1 (0 in the borders)
# convert this into a image, append it to an array. at the end of metro algo, we have the snapshots of all our monte carlo iterations.!
# convert this set of image arrays into a gif with appropriate fps
