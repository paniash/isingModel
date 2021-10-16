# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:10:19 2021

@author: gauta
"""
#defining energy fn for next to next atom interactions

#fn to take care of periodic index of our lattice
def pbc(i,d):
    if i<d:
        return i
    
    else:
        return d-i

#fn to return the energy of our lattice
def energy(l,d):
    e=0
    #center is the atom, left n right r interactions with neighbours
    j=[-0.5,-1,0,-1,-0.5]
    
    #our energy assumes periodic boundary conditions
    for i in range(l.size):
      
      e+=l[i]*(j[pbc(-2,d)]*l[pbc(i-2,d)]
                 +j[pbc(-1,d)]*l[pbc(i-1,d)]
                 +j[pbc(0,d)]*l[pbc(i,d)]
                 +j[pbc(1,d)]*l[pbc(i+1,d)]
                 +j[pbc(2,d)]*l[pbc(i+2,d)])
    
    return e