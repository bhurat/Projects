# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:29:58 2019

@author: bazzz
"""
import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt

np.random.seed(3929)
def IC(x):
    return (x[0]-55)**2 + (x[1]-75)**2 - 50

def imgNoise(A,x):
    w,h = A.shape
    Anew = np.zeros([w,h,3])
    Anew[:,:,0] = A+ x * np.random.normal(size=A.shape)
    Anew[:,:,1] = Anew[:,:,0]
    Anew[:,:,2] = Anew[:,:,0]
    return Anew/255.0
def nabla(Im):
    Im1 = Im[:,:,0]
    h,w = Im1.shape
    dIm = np.zeros([h,w,2])
    #dA_x = A(x+1,y) - A(x,y) (Border = 0)
    dIm[:-1,:,1] = Im1[1:,:] - Im1[:-1,:]
    #dIm_y = A(x,y+1) - A(x,y) (Border = 0)
    dIm[:,:-1,0] = Im1[:,1:] - Im1[:,:-1]
    return dIm

def dispImage(img,surf,paus = .5):
    plt.clf()
    plt.imshow(img)
    plt.contour(surf,0)
    plt.pause(paus)
    
def ForcingImg(img,lamb):
    dIm = nabla(img)
    absdIm = np.sqrt(dIm[:,:,0]**2 + dIm[:,:,1]**2)
    #ForcIm = 1./(1+lamb*absdIm)             #Lambda is high = strongest gradients only
    ForcIm = np.exp(-absdIm**2/(2*lamb**2)) #lambda is low = strongest gradients only
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
    
def gradFTFS(surf):
    dx = np.copy(surf)
    dx[:-1,:] -= surf[1:,:]
    dx[-1,:] = 255
    dy = np.copy(surf)
    dy[:,:-1] -= surf[:,1:]
    dy[:,-1] = 255
    return np.sqrt(dx**2 + dy**2)
def gradFTBS(surf):
    dx = np.copy(surf)
    dx[:-1,:] -= surf[1:,:]
    dx[-1,:] = 0
    dy = np.copy(surf)
    dy[:,:-1] -= surf[:,1:]
    dy[:,-1] = 0
    return np.sqrt(dx**2 + dy**2)

def UpwindingScheme(surf,F,lamb = .8):
    surf1 = np.copy(surf)    
    if F[x,y] >= 0:
       surf1 = surf - F*lamb*gradFTFS(surf)    
    else:
        surf1 = surf - F*lamb*gradFTBS(surf)
    return surf1
                
def UpwindingScheme2(surf,F,lamb = .8):
    surf1 = np.copy(surf)
    w,h = surf.shape
    surf1 = surf - F*lamb*np.sqrt((surf[2,1] - surf[1,1])**2 + (surf[1,2]-surf[1,1])**2)
    # for x in range(1,w):
    #     for y in range(1,h):
    #         if F[x,y] >= 0:
    #            surf1 = surf - F*lamb*np.sqrt((surf[x+1,y] - surf[x,y])**2 + (surf[x,y+1]-surf[x,y])**2)
    #         else:
    #             surf1 = surf - F*lamb*np.sqrt((surf[x,y] - surf[x-1,y])**2 + (surf[x,y]-surf[x,y-1])**2)
    #         return surf1
    return surf1
im1 = imread('test3.png',pilmode = 'L')


h,w,pix = np.shape(im1)
yh = np.arange(0,h)
xh = np.arange(0,w)
YY,XX = np.meshgrid(xh,yh)
surf = IC([YY,XX])
print(np.shape(im1))
print(surf)
F = ForcingImg(im1,.15)
F = regionForcing(im1,surf)

h = 1; lamb = .8
k = lamb*h

tIter = 100

tk = np.arange(0,(tIter+1)*k,k)
dispImage(im1,surf)

for i in tk:
    # surf = surf - lamb*(max(F,0)*dif+1 + min(F,0)*dif-1)
    surf = UpwindingScheme2(surf,F,lamb)
    F = regionForcing(im1[:,:,0],surf)
    dispImage(im1,surf,paus = .2)

#f_t + F* abs delta f   = 0
#f_t = -F abs(delta f)
#vn1 = vn + -F*k(vmp1 - vm) vm - vmm1

dispImage(im1,surf)

print("Done!")

