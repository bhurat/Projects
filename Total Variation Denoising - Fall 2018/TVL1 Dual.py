# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:25:05 2018

@author: bazzz
"""

import numpy as np
import scipy as sp
from math import sqrt
import matplotlib.pyplot as plt
from scipy.misc import face
import time
from imageio import imread, imsave


########### - PICK YOUR IMAGE - ############
#Z = face()[100:500,400:1000,1]/255
#imgName = 'Face'
#Z = imread('Rain.jpg')[:,:,0]/255
#imgName = 'Rain'
Z = imread('Bas.png')[:,:,0]/255
imgName = 'Bas'


np.random.seed(3929)
def imgNoise(A):
    Anew = A+ 0.1 * np.random.normal(size=A.shape)
    return Anew

def nabla(A):
    h,w = A.shape
    dA = np.zeros([h,w,2])
    #dA_x = A(x+1,y) - A(x,y) (Border = 0)
    dA[:-1,:,1] = A[1:,:] - A[:-1,:]
    #dA_y = A(x,y+1) - A(x,y) (Border = 0)
    dA[:,:-1,0] = A[:,1:] - A[:,:-1]
    return dA
    
def nablaT(G):
    h,w = G.shape[:2]
    I = np.zeros((h,w),G.dtype)
    I[:,:-1] -= G[:,:-1,0]
    I[:,1:] += G[:,:-1,0]
    I[:-1] -= G[:-1,:,1]
    I[1:] += G[:-1,:,1]
    return I


def anorm(x):
    return np.sqrt((x*x).sum(-1))

def TVL1Norm(X, Orig, clambda):
    TV = anorm(nabla(X)).sum()
    Edata = (np.abs(X-Orig)).sum()
    return clambda*TV + Edata


def project_nd(P, r):
    nP = np.maximum(1.0, anorm(P)/r)
    return P / nP[...,np.newaxis]
    
def shrink_1d(X, F, step):
    return X + np.clip(F - X, -step, step)

# setting step sizes and other params
L2 = 8.0                #gradient norm
tau = 0.02              #Step size
sigma = 1.0 / (L2*tau)  #Step size
clambda = 1         #Determines how intense the denoising is
iter_n = 101

X = imgNoise(Z)
Y = np.copy(X)
P = nabla(Y)
clock1 = time.clock()

for i in range(0,iter_n):
    P = project_nd( P + sigma*nabla(Y), 1.0 )
    Y = shrink_1d(Y - tau*nablaT(P), X, clambda*tau)

clock2 = time.clock()

print(clock2-clock1)

plt.gray()
plt.imshow(Z)
imsave('.\TVL1Dual\ Orig'+imgName+'.png',Z)
plt.show()


plt.gray()
plt.imshow(X)
imsave('.\TVL1Dual\ Noisy'+imgName+'.png',X)
plt.show()

plt.gray()
plt.imshow(Y)
imsave('.\TVL1Dual\ Denoi'+imgName+'.png',Y)
plt.show()
