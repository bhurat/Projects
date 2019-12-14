# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:40:10 2019

@author: bazzz
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

#IMAGE PROCESSING/DISPLAY FUNCTIONS
def dispImage(img,surf, paus = .5):
    plt.clf()
    plt.imshow(img, cmap='gray')
    plt.contour(surf,0,colors = 'red')
    plt.pause(paus)

def savImage(img,surf,filename,tIter):
    plt.clf()
    plt.imshow(img,cmap = 'gray')
    plt.contour(surf,0,colors = 'red')
    filtot = filename+string(tIter)+'.png'
    plt.savefig(filtot)

#VELOCITY FORCING TERM
def regionForcing(img,surf):   #homogeneity of a region
    regPos = np.where(surf > 0)     #positive region
    muPos = img[regPos]
    muPos = np.sum(muPos)/muPos.size #avg of positive region
    regNeg = np.where(surf < 0)     #negative region
    muNeg = img[regNeg]
    muNeg = np.sum(muNeg)/muNeg.size #avg of negative region
    F = np.zeros(surf.shape)
    F[regPos] = (img[regPos] - muPos)**2; F[regNeg] = (img[regNeg] - muNeg)**2
    return F

#CURVATURE FORCING TERM
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

def curvForcing(img,beta):
    grIm = gradIm(img)
    dgrIm = nabla(grIm)
    return beta*dgrIm

#UPWIND SCHEME W/ CURVATURE FUNCTIONS
def IC(x, tau, size):
    return (x[0]-tau[0])**2 + (x[1]-tau[1])**2 - size**2

def spaceDiff(surf, direction, axis):
    h,w = surf.shape
    surf1 = np.copy(surf)
    if direction == 'f':
        if axis == 'y':
            surf1[:-1,:] = surf[1:,:] - surf[:-1,:] #dsurf[x,y] = surf[x+1,y]- surf[x,y]
        elif axis == 'x':
            surf1[:,:-1] = surf[:,1:] - surf[:,:-1]
    elif direction == 'b':
         if axis == 'y':
            surf1[1:,:] = surf[1:,:] - surf[:-1,:]     # dsurf[x,y] = surf[x,y] - surf[x-1,y]
         elif axis == 'x':
            surf1[:,1:] = surf[:,1:] - surf[:,:-1]
    return surf1
    
def upwind(surf,var):
    return (np.maximum(0,spaceDiff(surf,'b',var)) +
            np.minimum(0,spaceDiff(surf,'f',var)))
    
def gradSurf(surf,Vn):
    phi_1 =  1.0*upwind(surf,'x')
    phi_2 =  1.0*upwind(surf,'y')
    grSurf = Vn*np.sqrt(phi_1**2 + phi_2**2)
    return grSurf

def curvSurf2(surf,epsilon = .005):
    phi_xx = spaceDiff(surf,'f','x') - spaceDiff(surf,'b','x')
    phi_yy = spaceDiff(surf,'f','y') - spaceDiff(surf,'b','y')
    phi_x = .5*spaceDiff(surf,'c','x')
    phi_y = .5*spaceDiff(surf,'c','y')
    phi_xy = .25*(spaceDiff(spaceDiff(surf,'c','x'),'c','y'))
    numer = phi_xx*phi_y**2 - 2.0*phi_y*phi_x*phi_xy + phi_yy*phi_x**2
    denom = (epsilon+phi_x**2 + phi_y**2)**(3.0/2.0)
    return numer/denom

def curvSurf(surf,sn):
    sk = sn[:,:,1]* upwind(surf,'x') +sn[:,:,0]*upwind(surf,'y')
    return sk


#LEVEL SET DRIVER FUNCTION
def levelSetRegion(imgname, center, size, sigma, b, tIter = 600):
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
    for i in tk:
        Vn = regionForcing(im2,surf)
        surf = surf - lamb*(gradSurf(surf,Vn) + .01*curvSurf(surf,Sn))
        # dispImage(im1,surf,paus = .0001)
        
    dispImage(im1,surf,paus = .0001)
        
    print("Done!")
    

print("-levelSetRegion(imgname, center, size, sigma, b, tIter = 600) usage-")
print(" imgname: 'example.png' \n center: [x,y]\n size: x\n blurring param sigma: x")
print(" curve forcing param b: x\n tIter: int")

# levelSetRegion('Test3.png',[55,55],2,.02,.02)
# levelSetRegion('MRI_Heart.png', [150,150], 5, .02, .02)