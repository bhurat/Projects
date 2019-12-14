# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:54:07 2019

@author: bazzz
"""

import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt
from math import exp
np.random.seed(3929)

def IC(x, tau, size):
    return (x[0]-tau[0])**2 + (x[1]-tau[1])**2 - size

def nabla(Im):      #2d b/w img only
    h,w = Im.shape
    dIm = np.zeros([h,w,2])
    dIm[:-1,:,1] = Im[1:,:] - Im[:-1,:]
    dIm[:,:-1,0] = Im[:,1:] - Im[:,:-1]
    return dIm

def gradForcing(Im,lamb):        #Gradient Based Forcing Term
    dIm = nabla(Im)
    absdIm = np.sqrt(dIm[:,:,0]**2 + dIm[:,:,1]**2)
    
    ### FORCING FUNCTION 1 : Lambda is high = strongest gradients only
    #ForcIm = 1./(1+lamb*absdIm)             
    
    ###FORCING FUNCTION 2 : lambda is low = strongest gradients only
    ForcIm = np.exp(-absdIm**2/(2*lamb**2)) 
    
    return ForcIm

def regionForcing(img,surf,kappa = 0):   #homogeneity of a region
    regPos = np.where(surf > 0)
    muPos = img[regPos]
    muPos = np.sum(muPos)/muPos.size
    regNeg = np.where(surf < 0)
    muNeg = img[regNeg]
    muNeg = np.sum(muNeg)/muNeg.size
    F = np.zeros(img.shape)
    F[regPos] = img[regPos] - muPos + kappa
    F[regNeg] = img[regNeg] - muNeg + kappa
    return F    
    

def dispImage(img,surf, paus = .5):
    plt.clf()
    plt.imshow(img)
    plt.contour(surf,0,colors = 'red')
    plt.pause(paus)
    
def gradForw(surf,F,choice = 1):
    if choice == 1:
        h,w = surf.shape
        gradsurf = np.zeros([h,w])
        for y in range(1,h-1):
            for x in range(1,w-1):
                if F[y,x] >= 0:
                    gradsurf[y,x] = F[y,x]*np.sqrt((surf[y+1,x] - surf[y,x])**2 + (surf[y,x+1]-surf[y,x])**2)
                else:
                    gradsurf[y,x] = F[y,x]*np.sqrt((surf[y,x] - surf[y-1,x])**2 + (surf[y,x]-surf[y,x-1])**2)
        return gradsurf
        
    
def gradBack(surf,x,y,choice = 1):
        h,w = surf.shape
        gradsurf = np.zeros([h,w])
        for y in range(1,h-1):
            for x in range(1,w-1):
                gradsurf[y,x] = np.sqrt((surf[y,x] - surf[y-1,x])**2 + (surf[y,x]-surf[y,x-1])**2)
        return gradsurf        
    
def upwindingScheme(surf,F,lamb = .8): 
    surf1 = np.copy(surf)
    
    surf1 = surf - lamb*gradForw(surf,F)
    return surf1

im1 = imread('Test1.png',pilmode = 'RGB')
im1 = im1.astype(np.float) / 255

h,w,pix = np.shape(im1)
xh = np.arange(0,w)
yh = np.arange(0,h)
XX,YY = np.meshgrid(xh,yh)
surf = IC([XX,YY],[75,75],50)

dispImage(im1,surf,paus = 0.1)
plt.figure()

h = 1; lamb = .8
k = lamb*h

timeIter = 200

dispImage(im1,surf)
tk = np.arange(0,(timeIter+1)*k,k)
for i in tk:
    # F = regionForcing(im1[:,:,0], surf, k = 0)
    F = gradForcing(im1[:,:,0],.1)
    surf = upwindingScheme(surf,F,lamb = .8)
    dispImage(im1,surf,paus = .1)
    
print("Done!")
