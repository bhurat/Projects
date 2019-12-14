# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:40:10 2019

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
    plt.imshow(img, cmap='gray')
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

def regionForcing(img,surf,kappa = 0):   #homogeneity of a region
    regPos = np.where(surf > 0)     #positive region
    muPos = img[regPos]
    muPos = np.sum(muPos)/muPos.size #avg of positive region
    regNeg = np.where(surf < 0)     #negative region
    muNeg = img[regNeg]
    muNeg = np.sum(muNeg)/muNeg.size #avg of negative region
    F = np.zeros(img.shape)
    F[regPos] = (img[regPos] - muPos)**2 + kappa
    F[regNeg] = (img[regNeg] - muNeg)**2 + kappa
    return F  

def levelSetRegion(imgname, center, size, sigma, a, b, tIter = 600):
    pp = a
    im1 = imread(imgname,pilmode = 'L')
    im1 = im1.astype(np.float) / 255
    im1 = im1/np.max(im1)
    im2 = gaussian_filter(im1,sigma)
    
    
    h,w = np.shape(im1)
    yh = np.arange(0,h)
    xh = np.arange(0,w)
    XX,YY = np.meshgrid(xh,yh)
    surf = IC([XX,YY],center,size)
    Vn = regionForcing(im2,surf,pp)
    Sn = curvForcing(im2,b)
    c = 1.0;
    h = 1.0; lamb = c/(np.max(np.abs(Vn)) + np.max(np.abs(Sn)))
    k = lamb*h
    
    tIter = tIter
    
    tk = np.arange(0,(tIter+1)*k,k)
    for i in tk:
        Vn = regionForcing(im2,surf,pp)
        surf = surf - lamb*(gradSurf(surf,Vn)+ curvSurf(surf,Sn))
        dispImage(im1,surf,paus = .0001)
        
    print("Done!")
    

print("-levelSetRegion(imgname, center, size, sigma, a, b, tIter = 600) usage-")
print(" imgname: 'example.png' \n center: [x,y]\n size: x\n blurring param sigma: x")
print(" region forcing param a: x \n curve forcing param b: x\n tIter: int")

# levelSetRegion('MRI_Heart.png', [150,150], 5, 0, .02, .02)