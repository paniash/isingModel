###create a random 40x40 array of up-down spin (for 2d ising lattice. 1d array for 1d lattice)
import numpy as np
import matplotlib.pyplot as plt

#dimension of our array
d=40

lattice=np.random.random((d,d))
#relabel 1 and -1 w.r.t the probability of getting it down=p
p=0.5
lattice[lattice>=p]=1
lattice[lattice<p]=-1

#create 0 in the border (just to make neighbour atom calculations easier)
lattice=np.pad(array=lattice, pad_width=1, mode='constant')

###define a function to calculate the total energy of our lattice
def get_energy(lattice):
    e=0
    #look at the i,jth atom. 
    for i in range(1,d):
        for j in range(1,d):
            #see the neighbouring atoms(up,down,left,right)
            g=np.array([[0,1,0],[1,0,1],[0,1,0]])
            
            #calculate the sigma_ij*sigma_i'j' around our atom at ij (stored in matrix form)
            m=-lattice[i,j]*g*lattice[i-1:i+2,j-1:j+2]
            
            #sum over all the neighbour sigmas
            e+=np.sum(m)
            # print(e,m,np.sum(m),i,j)
    
    return e

  
#just to check if our energy calculation makes sense for special cases:
#the most unstable energy(unalligned spins)
m=np.ones((d,d))
m[1::2,1::2]=-1
m[::2,::2]=-1
m=np.pad(array=m, pad_width=1, mode='constant')

#the most stable energy(alligned spins))
n=np.ones((d,d))
n=np.pad(array=n, pad_width=1, mode='constant')


print(get_energy(lattice),get_energy(m),get_energy(n))

#the ising model generated(green->up, red->down)
plt.imshow(lattice,cmap='prism')

#Implement metropolis algo (Source Wikipedia) // How reliable is this algo? Can we provide a guarantee of our datas quality?
#Pick a spin site using selection probability g(μ, ν) and calculate the contribution to the energy involving this spin.
#Flip the value of the spin and calculate the new contribution
#If the new energy is less, keep the flipped value.
#If the new energy is more, only keep with probability e − β ( H ν − H μ ) . {\displaystyle e^{-\beta (H_{\nu }-H_{\mu })}.} 
#Repeat

#we have the lattice configuration generated for our specified temperature β.

#generate as much data as you require for all sets of temperatures.
