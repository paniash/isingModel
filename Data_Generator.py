# optimised monte carlo by introducing classes energy functions and NUMBA...
import numpy as np
import numba
import time

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
d = 50
# optional addition of external magnetic field B
b = 0.0

# metropolis algorithm
@numba.njit("f8[:,:](f8[:,:], f8, i8)", nogil=True)
def metro(lattt, beta, reps):
    lat=lattt.copy()
    for i in range(reps):

        # random pick of our point in lattice
        x, y = np.random.randint(1, d), np.random.randint(1, d)

        # energies of our two systems
        #e1, e2 = flipcheck(lat, x, y)

        #manual spinflip check
        g = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        m = -lat[x, y] * g * lat[x - 1 : x + 2, y - 1 : y + 2]
        e1 = np.sum(m) - lat[x, y] * b
        lat[x, y] *= -1
        m = -lat[x, y] * g * lat[x - 1 : x + 2, y - 1 : y + 2]
        e2 = np.sum(m) - lat[x, y] * b
        lat[x, y] *= -1

        # if energy increase only keep with a probability(more energy increase, lesser the prob)
        if e2 > e1:
            # probability of flipping if unfavourable
            p = np.exp(-beta * (e2 - e1))

            if np.random.random() < p:

                lat[x, y] *= -1

        # flip the lattice spin if energy decreases
        else:
            lat[x, y] *= -1

    return lat

#temperature fineness
h=20
t=np.linspace(0.25,4,h)

Master=[]
for T in t:

    data=[]
    print('T: %s'%T)
    print('Data generated:')

    #no of datasets in each temperature
    n=10000
    mat=time.time()
    for i in range(n):
      print(i)
      start=time.time()
      lat = ising_model_2D(d, 0.5, b)

      lat.lattice=metro(lat.lattice, 1/T, 60000)
      lat.energy=lat.get_energy()
      lat.spin=np.sum(lat.lattice)

      data.append(lat)
      #plt.imshow(lat.lattice, cmap="Greys_r", interpolation="nearest", origin="lower")
      print('Execution time: ',time.time()-start)
    #save data in a file! Gets destroyed every iteration!!!
    data=np.array(data)
    Master.append(data)
    print('Total time for beta: ',time.time()-mat,'\n')

Master=np.array(Master)
np.save('2d_ising_data_2',Master,allow_pickle=True)
np.savetxt('Temperature.csv',t,delimiter=',')


#for loading the npy pickle files:
# np.load('2d_ising_data_2.npy',allow_pickle=True)
