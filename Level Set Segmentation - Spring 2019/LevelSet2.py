# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:24:30 2019

@author: bazzz
"""


import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt
from math import exp
from scipy.ndimage.filters import gaussian_filter
np.random.seed(3929)

def dispImage(img,surf, paus = .5):
    plt.clf()
    plt.imshow(img)
    plt.contour(surf,0,colors = 'red')
    plt.pause(paus)
    
def IC(x, tau, size):
    return (x[0]-tau[0])**2 + (x[1]-tau[1])**2 - size**2

def nabla(img):      #2d b/w img only
    h,w = img.shape
    dIm = np.ones([h,w,2])
    dIm[1:-1,:,0] = .5*(img[2:,:] - img[:-2,:]) #central y
    dIm[:,1:-1,1] = .5*(img[:,2:] - img[:,:-2]) #central x
    return dIm

def gradIm(img):
    dIm = nabla(img)
    grad = dIm[:,:,0]**2 + dIm[:,:,1]**2
    grad = np.sqrt(grad)
    return grad

def gradForcing(img,alph):
    grad = gradIm(img)
    F = np.exp(-alph* grad)
    return F
def gaussBlur(img,sigma):
    return 1/(2*np.pi*sigma**2)*np.exp(-img)
def spaceDiff(surf, direction, axis):
    h,w = surf.shape
    surf1 = np.copy(surf)
    if direction == 'forward':
        if axis == 'y':
            surf1[:-1,:] = surf[1:,:] - surf[:-1,:] #dsurf[x,y] = surf[x+1,y]- surf[x,y]
        elif axis == 'x':
            surf1[:,:-1] = surf[:,1:] - surf[:,:-1]
    elif direction == 'backward':
         if axis == 'y':
            surf1[1:,:] = surf[1:,:] - surf[:-1,:]     # dsurf[x,y] = surf[x,y] - surf[x-1,y]
         elif axis == 'x':
            surf1[:,1:] = surf[:,1:] - surf[:,:-1]
    return surf1
    
def upwind(surf,var):
    return np.maximum(0,spaceDiff(surf,'backward',var)) + np.minimum(0,spaceDiff(surf,'forward',var))
    
def gradSurf(surf,Vn):
    phi_1 = np.maximum(0,Vn)*spaceDiff(surf,'backward','x') + np.minimum(0,Vn)*spaceDiff(surf,'forward','x')
    phi_2 = np.maximum(0,Vn)*spaceDiff(surf,'backward','y') + np.minimum(0,Vn)*spaceDiff(surf,'forward','y')

    phi_1 = Vn* upwind(surf,'x')
    phi_2 = Vn* upwind(surf,'y')
    grSurf = np.sqrt(phi_1**2 + phi_2**2)
    return grSurf

def curvForcing(img,beta):
    grIm = gradIm(img)
    dgrIm = nabla(grIm)
    return beta*dgrIm

def curvSurf(surf,sn):
    sk = np.maximum(0,sn[:,:,0])*spaceDiff(surf,'backward','x') + np.minimum(0,sn[:,:,0])*spaceDiff(surf,'forward','x') + np.maximum(0,sn[:,:,1])*spaceDiff(surf,'backward','y') +  np.minimum(0,sn[:,:,1])*spaceDiff(surf,'forward','y')
    sk = sn[:,:,1]* upwind(surf,'x') +sn[:,:,0]*upwind(surf,'y')
    return sk

im1 = imread('test3.png',pilmode = 'RGB')
im2 = gaussian_filter(im1[:,:,0],1.0)
# im1 = im1.astype(np.float) / 255

h,w,pix = np.shape(im1)
yh = np.arange(0,h)
xh = np.arange(0,w)
XX,YY = np.meshgrid(xh,yh)
surf = IC([XX,YY],[25,25],15)

dispImage(im1,surf,paus = 0.1)

Vn = gradForcing(im2,.15)

Sn = curvForcing(im2,.01)
c = 1.0;
h = 1.0; lamb = c/(np.max(np.abs(Vn)) + np.max(np.abs(Sn)))
k = lamb*h

tIter = 350

tk = np.arange(0,(tIter+1)*k,k)
for i in tk:
    
    surf = surf - lamb*(gradSurf(surf,Vn)+ curvSurf(surf,Sn))
    dispImage(im1,surf,paus = .0001)
    
print("Done!")