# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:03:09 2019

@author: bazzz
"""
import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt

def imgNoise(A,x):
    w,h = A.shape
    Anew = np.zeros([w,h,3])
    Anew[:,:,0] = A+ x * np.random.normal(size=A.shape)
    Anew[:,:,1] = Anew[:,:,0]
    Anew[:,:,2] = Anew[:,:,0]
    return Anew/255.0
xx = imread('Test2.png',pilmode = 'L')
yy = imgNoise(xx,15.)
plt.imshow(yy,cmap = 'gray')
#imsave('Test3.png',yy)
