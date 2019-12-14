# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:31:15 2018

@author: bazzz
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.misc import face
import time
from imageio import imread, imsave

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
def ROF(X, Orig, clambda):
    TV = anorm(nabla(X)).sum()
    Edata = ((X - Orig)**2).sum()
    return TV + clambda*Edata/2

#TVL1
def TVL1(X, Orig, clambda):
    TV = anorm(nabla(X)).sum()
    Edata = (np.abs(X-Orig)).sum()
    return clambda*TV + Edata

filename = ['TVL1Steepest', 'ROFSteepest', 'ROFDual', 'TVL1Dual']
imgName = ['Face']
for j in filename:
    for i in imgName:
        X = imread('.\\'+j+'\ Noisy'+i+'.png')
        
        Y = imread('.\\'+j+'\ Denoi'+i+'.png')
        print(f'{j} and {i} - ROF Norm: {ROF(Y/255,X/255,8)}')
        print(f'{j} and {i} - TVL1 Norm: {TVL1(Y/255,X/255,1)}')

for i in imgName:
    X = imread('.\ROFDual\ Noisy'+i+'.png')
    print(f'Noisy {i} - ROF Norm: {ROF(X/255,X/255,8)}')
    print(f'Noisy {i} - TVL1 Norm: {TVL1(X/255,X/255,1)}')