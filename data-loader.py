import numpy as np
import matplotlib.pyplot as plt

# define an ising model class for loading the ising model class...
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

#2d ising data load
data=np.load('2d_ising_data_2.npy',allow_pickle=True)

#temperature label load
t=np.genfromtxt('temperature.csv', delimiter=',')

#to see the statistical results of our generated data
e,spin=np.zeros(20),np.zeros(20)

print("Size of Temperature data: ",t.size,'\n')
print("Size of 2D Ising Model data: ",data.shape,
      "\n No.of temperature samples taken: ",data.shape[0],
      "\n No.of Lattice per temperature sample: ",data.shape[1])

#plot the energy specific heat graffs
h=t.size
n=data.shape[1]

for i in range(h):
    avg_e=0
    avg_spin=0

    for j in range(n):
        avg_e+=data[i][j].energy
        avg_spin+=data[i][j].spin
    avg_e/=n
    avg_spin/=n

    e[i]=avg_e
    spin[i]=avg_spin


plt.figure()
plt.grid()
plt.title('Avg. Energy vs Temperature')
plt.xlabel('T (units)')
plt.ylabel('E (units)')
plt.plot(t,e)
plt.plot(t,e,'k .')
plt.show()

c=np.zeros(h)
r=1

for i in range(r,h-r):
    c[i]=(e[i+r]-e[i-r])/(t[i+r]-t[i-r])

plt.figure()
plt.grid()
plt.title('Heat capacity of lattice vs Temperature')
plt.xlabel('T (units)')
plt.ylabel('dE/dT (units)')
plt.plot(t[r:h-r],c[r:h-r])
plt.plot(t[r:h-r],c[r:h-r],'k .')
plt.show()
