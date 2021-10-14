# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:32:39 2021

@author: Lenovo
"""
#1d ising model at a very hi temperature
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#fn to return the energy of our lattice
def energy(l):
    e=0
    j=1
    #our energy assumes periodic boundary conditions
    for i in range(l.size-1):
        
        e+=-j*l[i]*(l[i-1]+l[i+1])
    
    e+=-j*l[l.size-1]*(l[l.size-2]+l[0])
    
    return e

#length of our lattice
d=50
#no of 1d ising models
#n=2440
n=10000
score_tr=[]
score_te=[]

#defining our lattice
l=np.random.choice([-1, 1], size=(n,d))

#array to store the labels
label=np.zeros(n)

#creating the labels
for i in range(n):
  label[i]=energy(l[i])

#partition of our training n testing sets
m=int(7*n/10)

train=l[:m]
test=l[m:]

train_label=label[:m]
test_label=label[m:]

#data generated!!

#setting up our ML algos
leastsq=linear_model.LinearRegression()
ridge=linear_model.Ridge(tol=0.01)
lasso = linear_model.Lasso()

#needed for training the data: computing the outer product of S_i
train_states=np.einsum('...i,...j->...ij', train, train)
shape=train_states.shape

#each row is a reshaped from a 5x5 matrix of S_i*S_j
train_states=train_states.reshape((shape[0],shape[1]*shape[2]))

#needed for training the data: computing the outer product of S_i
test_states=np.einsum('...i,...j->...ij', test, test)
shape=test_states.shape

#each row is a reshaped from a 5x5 matrix of S_i*S_j
test_states=test_states.reshape((shape[0],shape[1]*shape[2]))

l=np.logspace(-5,6,100)
l=np.round(l,5)
i=0
for a in l:
    i+=1
    
    #training our dataset
    ridge.set_params(alpha=a)
    ridge.fit(train_states, train_label)
    J=ridge.coef_.reshape((shape[1],shape[2]))
    
    mark1=ridge.score(train_states,train_label)*100
    mark2=ridge.score(test_states,test_label)*100
    
    score_tr.append(mark1)
    score_te.append(mark2)
    
    # print("r'$\lambda$: %s"%a,"train-test: ",m,'-',n-m)
    # print("Training score: %s /100"%mark1)
    # print("Testing score: %s /100"%mark2,'\n')
    
    plt.figure()
    plt.title(r'$\lambda$=%s'%a)
    c=plt.imshow(J,vmin=-1, vmax=1,cmap='Spectral_r',interpolation='nearest', origin='lower')
    plt.colorbar(c)
    plt.savefig(r'C:\Users\gauta\Downloads\Ridge lambda variation\ridgeJ-%s.png'%i)


plt.figure()
plt.grid()
plt.title(r'Variation of train and test score. n: %s'%n)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$R^{2}$ (in %)')
plt.plot(np.log(l),score_tr,label='Training Score')
plt.plot(np.log(l),score_te,label='Testing Score')
plt.legend()
plt.show()
