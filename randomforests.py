# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 18:22:05 2021

@author: gauta
"""

import numpy as np
import matplotlib.pyplot as plt

import time

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


# 2d ising data load
data=np.load('2d_ising_data_2.npy',allow_pickle=True)

# temperature label load
t=np.genfromtxt('temperature.csv', delimiter=',')

for i in range(20):
    for j in range(10000):
        data[i][j].lattice = data[i][j].lattice[1:-1,1:-1].flatten()

data_ordered = data[:8]
data_critical = data[8:14]
data_disordered = data[14:]


# our training data
X_ordered = []
for i in range(len(data_ordered[:])):
    # lattice very close to critical temperature
    if i==len(data_ordered[:])-1:
        for j in range(data_ordered[0].size):
            X_ordered.append(data_ordered[i][j].lattice)
            X_ordered.append(data_ordered[i][j].lattice)
            X_ordered.append(data_ordered[i][j].lattice)
    # lattice bit near to critical temperature
    elif i==len(data_ordered[:])-2:
        for j in range(data_ordered[0].size):
            X_ordered.append(data_ordered[i][j].lattice)
            X_ordered.append(data_ordered[i][j].lattice)
    # lattice bit far from critical temperature
    else:
        for j in range(data_ordered[0].size):
            X_ordered.append(data_ordered[i][j].lattice)

X_disordered = []
for i in range(len(data_disordered[:])):
    # lattice very close to critical temperature
    if i==len(data_disordered[:])-1:
        for j in range(data_disordered[0].size):
            X_disordered.append(data_disordered[i][j].lattice)
            X_disordered.append(data_disordered[i][j].lattice)
            X_disordered.append(data_disordered[i][j].lattice)
    # lattice bit near to critical temperature
    elif i==len(data_disordered[:])-2:
        for j in range(data_disordered[0].size):
            X_disordered.append(data_disordered[i][j].lattice)
            X_disordered.append(data_disordered[i][j].lattice)
    # lattice bit far from critical temperature
    else:
        for j in range(data_disordered[0].size):
            X_disordered.append(data_disordered[i][j].lattice)

X_critical = []
for i in range(len(data_critical[:])):
    for j in range(data_critical[0].size):
        X_critical.append(data_critical[i][j].lattice)


del data, data_ordered, data_disordered, data_critical

# data labels for our data
y_ordered, y_critical, y_disordered = [], [], []

for i in range(len(X_ordered)):
    y_ordered.append(1)

for i in range(len(X_disordered)):
    y_disordered.append(0)

for k in range(len(X_critical)//2):
    y_critical.append(1)

for k in range(len(X_critical) - len(X_critical)//2):
    y_critical.append(0)

X = np.concatenate((X_ordered, X_disordered))
y = np.concatenate((y_ordered, y_disordered))

print(len(y_critical))

from sklearn.model_selection import train_test_split
train_to_test_ratio = 0.8

X_train, X_test, y_train, y_test = train_test_split(X, y,
        train_size=train_to_test_ratio, test_size=1.0-train_to_test_ratio)

print('X_train shape:', X_train.shape)
print('Y_train shape:', y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(len(X_critical), 'critical samples')
print(X_test.shape[0], 'test samples')

#%% Training model
from sklearn.ensemble import RandomForestClassifier

min_estimators = 10
max_estimators = 201
classifer = RandomForestClassifier

#%%
n_estimator_range = np.arange(min_estimators, max_estimators, 10)
leaf_size_list =[10,2500,5000,7500,10000]

m = len(n_estimator_range)
n = len(leaf_size_list)

RFC_OOB_accuracy=np.zeros((n,m))
RFC_train_accuracy=np.zeros((n,m))
RFC_test_accuracy=np.zeros((n,m))
RFC_critical_accuracy=np.zeros((n,m))
run_time=np.zeros((n,m))

print_flag = True

for i, leaf_size in enumerate(leaf_size_list):
    # Define Random Forest Classifier
    myRF_clf = classifer(
        n_estimators=min_estimators,
        max_depth=None,
        min_samples_split=leaf_size, # minimum number of sample per leaf
        oob_score=True,
        random_state=0,
        warm_start=True # this ensures that you add estimators without retraining everything
    )
    for j, n_estimator in enumerate(n_estimator_range):

        print('n_estimators: %i, leaf_size: %i'%(n_estimator,leaf_size))

        start_time = time.time()
        myRF_clf.set_params(n_estimators=n_estimator)
        myRF_clf.fit(X_train, y_train)
        run_time[i,j] = time.time() - start_time

    # check accuracy
        RFC_train_accuracy[i,j]=myRF_clf.score(X_train,y_train)
        RFC_OOB_accuracy[i,j]=myRF_clf.oob_score_
        RFC_test_accuracy[i,j]=myRF_clf.score(X_test,y_test)
        RFC_critical_accuracy[i,j]=myRF_clf.score(X_critical,y_critical)
        if print_flag:
            result = (run_time[i,j], RFC_train_accuracy[i,j], RFC_OOB_accuracy[i,j], RFC_test_accuracy[i,j], RFC_critical_accuracy[i,j])
            print('{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}'.format("time (s)","train score", "OOB estimate","test score", "critical score"))
            print('{0:<15.4f}{1:<15.4f}{2:<15.4f}{3:<15.4f}{4:<15.4f}'.format(*result))

#%%
plt.figure()
plt.title('Training Score')
plt.grid()
for i in range(n):
    plt.plot(n_estimator_range,RFC_train_accuracy[i],'--',label='Samples each node-%s'%leaf_size_list[i])
    plt.plot(n_estimator_range,RFC_train_accuracy[i],'.k')


plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Accuracy')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

plt.figure()
plt.title('Testing Score')
plt.grid()
for i in range(n):
    plt.plot(n_estimator_range,RFC_test_accuracy[i],'--',label='Samples each node-%s'%leaf_size_list[i])
    plt.plot(n_estimator_range,RFC_test_accuracy[i],'.k')

plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Accuracy')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

plt.figure()
plt.title('Critical Score')
plt.grid()
for i in range(n):
    plt.plot(n_estimator_range,RFC_critical_accuracy[i],'--',label='Samples each node-%s'%leaf_size_list[i])
    plt.plot(n_estimator_range,RFC_critical_accuracy[i],'.k' )


plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Accuracy')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
