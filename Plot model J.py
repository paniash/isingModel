# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 12:35:39 2021

@author: gauta
"""
d=50
# J=np.diag(,k=-1)
J=np.zeros((d,d))
j=[-0,-1,0,-1,-0]


for i in range(d):
    
    if i+2>=d:
        i=i-d
        
    J[i][i]=j[2]
    J[i][i+1]=j[3]
    J[i][i-1]=j[1]
    J[i][i+2]=j[4]
    J[i][i-2]=j[0]
    
# print(J)
plt.figure()
plt.title('J : model')
c=plt.imshow(J,vmin=-1, vmax=1,cmap='Spectral_r',interpolation='nearest', origin='lower')
plt.colorbar(c)
plt.savefig(r'G:\Sem 7 (labs+online)\ML\Ising model\Lasso reg_1D\lassoJ-%s.png'%i)
    