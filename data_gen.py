###create a random 40x40 array of up-down spin (for 2d ising lattice. 1d array for 1d lattice)
import numpy as np
import matplotlib.pyplot as plt

#dimension of our array
d=40

lattice=np.random.random((d,d))
#lattice with a:(1-a) down-up
a=0.8

lattice[lattice>=a]=1
lattice[lattice<a]=-1

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

#the ising model generated(white->up, black->down)
plt.imshow(lattice,cmap='Greys_r',interpolation='nearest', origin='lower')

###Implement metropolis algo (Source Wikipedia) // How reliable is this algo? Can we provide a guarantee of our datas quality?
#metropolis algo
def metro(lattice,reps):
    #our inverse temperature
    beta=.5
    
    #for plotting our monte carlo data
    iterate,energy,spin=np.zeros(reps),np.zeros(reps),np.zeros(reps)
    
    for i in range(reps):
        
        #random pick of our point in lattice
        x,y=np.random.randint(1,d),np.random.randint(1,d)
        
        #make a copy of lattice with the spin at our random point flipped
        lattice_flip=lattice.copy()
        lattice_flip[x,y]*=-1
        
        #energies of our two systems
        e1=get_energy(lattice)
        e2=get_energy(lattice_flip)
        
        #if energy increase only keep with a probability(more energy increase, lesser the prob)
        if e2>e1:
            #probability of flipping if unfavourable
            p=np.exp(-beta*(e2-e1))
            
            if np.random.random()<p:
                
                lattice[x,y]*=-1
            
        #flip the lattice spin if energy decreases
        else:
            lattice[x,y]*=-1
        
        #dont need the copy after comparison
        del lattice_flip
        
        #save the energy
        energy[i]=get_energy(lattice)
        iterate[i]=i
        spin[i]=np.sum(lattice)
    return iterate,energy,spin

#we have the lattice configuration generated for our specified temperature β.
x,y,z=metro(lattice,10)
plt.figure()
plt.grid()
plt.plot(x,y)
plt.plot(x,z)
plt.show()    

plt.imshow(lattice,cmap='Greys_r',interpolation='nearest', origin='lower')

#generate as much data as you require for all sets of temperatures.

#try out animating the plots as monte carlo runs..?
