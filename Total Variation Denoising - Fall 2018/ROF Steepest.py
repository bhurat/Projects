# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:22:49 2018

@author: bazzz
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.misc import face
import time
from imageio import imread, imsave

Z = face()[100:500,400:1000,1]/255
imgName = 'Face'
#Z = imread('Rain.jpg')[:,:,0]/255
#imgName = 'Rain'
#Z = imread('Bas.png')[:,:,0]/255
#imgName = 'Bas'

np.random.seed(3929)
def imgNoise(A):
    Anew = A+ 0.1 * np.random.normal(size=A.shape)
    return Anew

def anorm(x):
    return np.sqrt((x*x).sum(-1))

def nabla(A):
    h,w = A.shape
    dA = np.zeros([h,w,2])
    #dA_x = A(x+1,y) - A(x,y) (Border = 0)
    dA[:-1,:,1] = A[1:,:] - A[:-1,:]
    #dA_y = A(x,y+1) - A(x,y) (Border = 0)
    dA[:,:-1,0] = A[:,1:] - A[:,:-1]
    return dA
    
#ROF NORM
def g1(X, Orig, clambda):
    TV = anorm(nabla(X)).sum()
    Edata = ((X - Orig)**2).sum()
    return TV + clambda*Edata/2

def gradg_ij(Y,i,j,X,clambda,alph = .001):
    padY[1:-1,1:-1] = Y
    return clambda*(Y[i,j] - X[i,j]) + (
            (2*padY[i+1,j+1]-padY[i+2,j+1]-padY[i+1,j+2])/sqrt(alph+(padY[i+2,j+1]-padY[i+1,j+1])**2+(padY[i+1,j+2]-padY[i+1,j+1])**2)+
            (padY[i+1,j+1]-padY[i,j+1])/sqrt(alph+(padY[i+1,j+1]-padY[i,j+1])**2 + (padY[i,j+2]-padY[i,j+1])**2)+
            (padY[i+1,j+1]-padY[i+1,j])/sqrt(alph + (padY[i+2,j]-padY[i+1,j])**2 + (padY[i+1,j+1]-padY[i+1,j])**2))

def gradg(Y,X,clambda,alph = .0001):
    for i in range(0,h):
        for j in range(0,w):
            gradY[i,j] = gradg_ij(Y,i,j,X,clambda,alph)
    return gradY

Y = imgNoise(Z)

X = Y

padY = np.pad(Y,1,'constant')

h,w = Y.shape
gradY = np.zeros((h,w))
clock1 = time.clock()
diff = 1E-8
clambda = 8
rho = 0.5
c1 = 10**(-4)
gr = gradg(Y,X,clambda)
counter = 0
grOld = np.zeros((h,w))
while (abs(np.linalg.norm(gr)-np.linalg.norm(grOld))> diff and counter < 150):
    grOld = np.copy(gr)
    counter += 1
    pk = -gr
    alph = 1
    while (g1(Y+alph*pk,X,clambda) > g1(Y,X,clambda) + c1*alph*np.reshape(pk,pk.size).dot(np.reshape(gr,gr.size))): #Backtracking Algo
        alph *= rho    
    Y = Y + alph*pk
    gr = gradg(Y,X,clambda)
    print(f'{counter}, : , {np.linalg.norm(gr)}')
    
clock2 = time.clock()
print(clock2-clock1)
plt.gray()
plt.imshow(Z)
imsave('.\ROFSteepest\ Orig'+imgName+'.png',Z)
plt.show()

plt.gray()
plt.imshow(X)
imsave('.\ROFSteepest\ Noisy'+imgName+'.png',X)
plt.show()

plt.gray()
plt.imshow(Y)
imsave('.\ROFSteepest\ Denoi'+imgName+'.png',Y)
plt.show()
