# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:28:49 2019

@author: bazzz
"""
import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
np.random.seed(3929)

#IMAGE PROCESSING/DISPLAY FUNCTIONS
def dispImage(img,surf,t,a,b, paus = .5):       #Displays image
    plt.clf()
    plt.imshow(img, cmap='gray')
    plt.contour(surf,0,colors = 'red')
    plt.title(f't = {t}, a = {a}, b = {b}')
    plt.pause(paus)

def savImage(img,surf,filename,tIter):          #Saves image to a file
    plt.clf()
    plt.imshow(img,cmap = 'gray')
    plt.contour(surf,0,colors = 'red')
    filtot = filename+str(tIter)+'.png'
    plt.savefig(filtot)

#VELOCITY FORCING TERM
    

def gradForcing(img,alph):  #Takes image, calculates gradient, and returns
    grad = gradIm(img)      #Gradient forcing term
    F = np.exp(-alph* grad)
    return F


#CURVATURE FORCING TERM
def nabla(img):      #2d b/w img only       NABLA: Takes horizontal and vertical 
    h,w = img.shape                       # central difference of image
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

#UPWIND SCHEME W/ CURVATURE FUNCTIONS
def IC(x, tau, size):    #Initial Condition - Paraboloid center at tau of radius size
    return (x[0]-tau[0])**2 + (x[1]-tau[1])**2 - size**2

def spaceDiff(surf, direction, axis):   #Function to take forward or backwards 
    h,w = surf.shape                    #difference of specified variable
    surf1 = np.copy(surf)
    if direction == 'f':    #Forward diff
        if axis == 'y':
            surf1[:-1,:] = surf[1:,:] - surf[:-1,:] 
        elif axis == 'x':
            surf1[:,:-1] = surf[:,1:] - surf[:,:-1]
    elif direction == 'b':  #Backward diff
         if axis == 'y':
            surf1[1:,:] = surf[1:,:] - surf[:-1,:]     
         elif axis == 'x':
            surf1[:,1:] = surf[:,1:] - surf[:,:-1]
    return surf1
    
def upwind(surf,var):           #Upwind scheme
    return (np.maximum(0,spaceDiff(surf,'b',var))+ np.minimum(0,spaceDiff(surf,'f',var)))
    
def gradSurf(surf,Vn):      #Nonlinear term |\nabla \phi|
    phi_1 = 1.0*upwind(surf,'x')
    phi_2 = 1.0*upwind(surf,'y')
    grSurf = Vn*np.sqrt(phi_1**2 + phi_2**2)
    return grSurf

def curvSurf(surf,sn):      #Curvature term
    sk = sn[:,:,1]* upwind(surf,'x') +sn[:,:,0]*upwind(surf,'y')
    return sk


#LEVEL SET DRIVER FUNCTION
def levelSetGrad(imgname, center, size, sigma, a, b, tIter = 600):
    im1 = imread(imgname,pilmode = 'L')
    im1 = im1.astype(np.float) / 255
    im1 = im1/np.max(im1)
    im2 = gaussian_filter(im1,sigma) #Gaussian Blur
    
    
    h,w = np.shape(im1)
    yh = np.arange(0,h)
    xh = np.arange(0,w)
    XX,YY = np.meshgrid(xh,yh)
    surf = IC([XX,YY],center,size)  #Set IC
    Vn = gradForcing(im2,a)
    Sn = curvForcing(im2,b)
    c = 1.0;
    h = 1.0; lamb = c/(np.max(np.abs(Vn)) + np.max(np.abs(Sn)))     #CFL Condition
    k = lamb*h
    
    tIter = tIter
    dispImage(im1,surf,0,a,b,paus = .0001)
    tk = np.arange(0,(tIter+1)*k,k)
    for i in range(0,len(tk)):
        surf = surf - lamb*(gradSurf(surf,Vn) - curvSurf(surf,Sn))
        if i%2 == 0:
            dispImage(im1,surf,i,a,b,paus = .0001)
            # savImage(im1,surf,'Title/Title_',i)   #Uncomment if you want to save results
    dispImage(im1,surf,i,a,b,paus = .0001)
    print("Done!")
    
print("-levelSetGrad(imgname, center, size, sigma, a, b, tIter = 600) usage-")
print(" imgname: 'example.png' \n center: [x,y]\n size: x\n blurring param sigma: x")
print(" grad forcing param a: x (small means faster, likelier to bleed)")
print("curve forcing param b: x\n tIter: int")

#POSSIBLE TESTS TO RUN (Uncomment)
# levelSetGrad('test2.png',[48,56],5,0,25,.45,100) 
# levelSetGrad('test3.png',[48,56],5,0,25,.25,1000) 
# levelSetGrad('test3.png',[48,56],5,0.5,25,.25,1000) 
# levelSetGrad('MRI_Heart.png', [150,150],5, .95, 125, .35,600)