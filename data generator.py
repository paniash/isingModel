#create a random 40x40 array of up-down spin (for 2d ising lattice. 1d array for 1d lattice)

#define a function to aclculate the total energy of our lattice

#Implement metropolis algo (Source Wikipedia) // How reliable is this algo? Can we provide a guarantee of our datas quality?
#Pick a spin site using selection probability g(μ, ν) and calculate the contribution to the energy involving this spin.
#Flip the value of the spin and calculate the new contribution
#If the new energy is less, keep the flipped value.
#If the new energy is more, only keep with probability e − β ( H ν − H μ ) . {\displaystyle e^{-\beta (H_{\nu }-H_{\mu })}.} 
#Repeat

#we have the lattice configuration generated for our specified temperature β.

#generate as much data as you require for all sets of temperatures.
