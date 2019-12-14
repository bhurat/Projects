# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:47:36 2019

@author: bazzz
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

#IMAGE PROCESSING/DISPLAY FUNCTIONS
def dispImage(img,surf, paus = .5):     #Displays Image
    plt.clf()
    plt.imshow(img, cmap='gray')
    plt.contour(surf,0,colors = 'red')
    plt.pause(paus)

def savImage(img,surf,filename,tIter):  #Saves image to a file
    plt.clf()
    plt.imshow(img,cmap = 'gray')
    plt.contour(surf,0,colors = 'red')
    filtot = filename+str(tIter)+'.png'
    plt.savefig(filtot)

def imThresh(img,thresh,plus = 20,minus = 20):      #Pre-processing Thrsholding
    yy = np.zeros(img.shape)
    yy += img
    cases = img < thresh
    yy[cases] -= minus
    yy[np.invert(cases)] += plus
    yy[yy>255] = 255
    yy[yy<0] = 0
    return yy

#VELOCITY FORCING TERM
def regionForcing(img,surf):   #homogeneity of a region
    regPos = np.where(surf > 0)     #positive region
    muPos = img[regPos]
    muPos = np.sum(muPos)/muPos.size #avg of positive region
    regNeg = np.where(surf < 0)     #negative region
    muNeg = img[regNeg]
    muNeg = np.sum(muNeg)/muNeg.size #avg of negative region
    F = np.zeros(img.shape)
    F[regPos] = (img[regPos] - muPos); F[regNeg] = (img[regNeg] - muNeg)
    return F  

#CURVATURE FORCING TERM
def nabla(img):      #2d b/w img only    NABLA: Takes horizontal and vertical 
    h,w = img.shape                    # central difference of image
    dIm = np.ones([h,w,2])
    dIm[1:-1,:,0] = .5*(img[2:,:] - img[:-2,:]) #central y
    dIm[:,1:-1,1] = .5*(img[:,2:] - img[:,:-2]) #central x
    return dIm

def gradIm(img):        #Takes the central differences of image and returns norm of gradient
    dIm = nabla(img)
    grad = dIm[:,:,0]**2 + dIm[:,:,1]**2
    grad = np.sqrt(grad)
    return grad

def curvForcing(img,beta):      #Calculates the Surface forcing term S
    grIm = gradIm(img)
    dgrIm = nabla(grIm)
    return beta*dgrIm

#LAX FRIEDRICH FLUX SCHEME W/ CURVATURE FUNCTIONS
def IC(x, tau, size):   #Initial Condition - Paraboloid center at tau of radius size
    return (x[0]-tau[0])**2 + (x[1]-tau[1])**2 - size**2

def spaceDiff(surf, direction, axis):   #Function to take forward, central or backwards 
    h,w = surf.shape                    #difference of specified variable
    surf1 = np.copy(surf)
    if direction == 'f': #Forward diff
        if axis == 'y':
            surf1[:-1,:] = surf[1:,:] - surf[:-1,:] 
        elif axis == 'x':
            surf1[:,:-1] = surf[:,1:] - surf[:,:-1]
    elif direction == 'b': #Backwards Diff
         if axis == 'y':
            surf1[1:,:] = surf[1:,:] - surf[:-1,:] 
         elif axis == 'x':
            surf1[:,1:] = surf[:,1:] - surf[:,:-1]
    elif direction == 'c': #Central Diff
         if axis == 'y':
             surf1[1:-1,:] = surf[2:,:] - surf[:-2,:]   
         elif axis == 'x':
             surf1[:,1:-1] = surf[:,2:] - surf[:,:-2]
    return surf1
    
def Hamilton(x,y,Vn):       #Hamilton Jacobi Equation
    return Vn*np.sqrt(x**2 + y**2)

def laxFriedFlux(surf,Vn,alphx = 1., alphy = 1.):   #LAX FRIEDRICHS FLUX SCHEME
    Ham = Hamilton(.5*(spaceDiff(surf,'b','x')+spaceDiff(surf,'f','x')),
                   .5*(spaceDiff(surf,'b','y') +spaceDiff(surf,'f','y')),Vn)
    return (Ham - .5*alphx*(spaceDiff(surf,'f','x') - spaceDiff(surf,'b','x')) 
            - .5*alphy*(spaceDiff(surf,'f','y') - spaceDiff(surf,'b','y')))

def upwind(surf,var):   #Upwind scheme (For curvature)
    return (np.maximum(0,spaceDiff(surf,'b',var)) +
            np.minimum(0,spaceDiff(surf,'f',var)))

def curvSurf(surf,sn):  #Curvature term
    sk = sn[:,:,1]* upwind(surf,'x') +sn[:,:,0]*upwind(surf,'y')
    return sk


#LEVEL SET DRIVER FUNCTION
def LFFluxRegion(imgname, center, size, sigma,  b, tIter = 600, xthresh = 1, ythresh = 1):
    im1 = imread(imgname,pilmode = 'L')
    im1 = im1.astype(np.float) / 255
    im1 = im1/np.max(im1)
    im2 = gaussian_filter(im1,sigma)
    
    
    h,w = np.shape(im1)
    yh = np.arange(0,h)
    xh = np.arange(0,w)
    XX,YY = np.meshgrid(xh,yh)
    surf = IC([XX,YY],center,size)
    Vn = regionForcing(im2,surf)
    Sn = curvForcing(im2,b)
    c = 1.0;
    h = 1.0; lamb = c/(np.max(np.abs(Vn)) + np.max(np.abs(Sn)))
    k = lamb*h
    
    tIter = tIter
    
    tk = np.arange(0,(tIter+1)*k,k)
    for i in range(0,len(tk)):
        Vn = regionForcing(im2,surf)
        surf = surf - lamb*(laxFriedFlux(surf,Vn,xthresh,ythresh) + curvSurf(surf,Sn))
        
        if i%4 == 0:
            # savImage(im1,surf,'MRI_Brain_3/MRI_Brain_3_',i)
            dispImage(im1,surf,paus = .0001)
    dispImage(im1,surf,paus = .0001)
    print("Done!")
    

print("-LFFluxRegion(imgname, center, size, sigma, b, tIter = 600) usage-")
print(" imgname: 'example.png' \n center: [x,y]\n size: x\n blurring param sigma: x")
print("curve forcing param b: x\n tIter: int\n") 
print("xthresh: x \n ythresh:y (Lower means more selective)")   

#POSSIBLE TESTS TO RUN (Uncomment)
# LFFluxRegion('Test3.png',[80,20], 5,0.,.05,350,.1,.1)
# LFFluxRegion('Brain2_2.png',[81,21],5,0, .1,75,.1,.1)