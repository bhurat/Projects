# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:01:52 2019

@author: bazzz
"""
import numpy as np
from sympy import nextprime

def legendre(n,p):
    K = pow(n,(p-1)//2,p)       #pow(a,b,c) is modular exponentiation
    return K
def Q(x, n):
    return (pow(x,2) - n)

def tonelli(n, p): #tonelli-shanks to solve modular square root, x^2 = N (mod p)
    assert legendre(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        r = pow(n, (p + 1) // 4, p)
        return r,p-r
    for z in range(2, p):
        if p - 1 == legendre(z, p):
            break
    c = pow(z, q, p)
    r = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        for i in range(1, m):
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i

    return (r,p-r)
        
        
def findSmooth(yy,base):
    xx = np.zeros([len(yy),len(base)+1]).astype(int)
    xx[:,0] = yy
    for p in range(0,len(base)):
        xxold = np.copy(xx[:,0])+1
        while (xx[:,0] != xxold).any():
            xxold = np.copy(xx[:,0])
            xx[xx[:,0]%base[p] == 0,p+1] += 1
            xx[xx[:,0]%base[p] == 0,0] //= base[p]
    return xx

def buildFactorBase(n,B):
    factorBase = np.arange(0,B)
    k = 0
    p = 2
    while(k < B):
        if legendre(n,p) == 1:
            factorBase[k] = p
            k += 1
        p = nextprime(p)
    return factorBase
        
def sieve(n,primes,xi,Qi):
    Qi1 = np.copy(Qi)
    k = 0
    if primes[0] == 2:
        k = 1
        i = 0
        while Qi1[i] % 2 != 0:
            i += 1
        for j in range(i,len(Qi1),2):
            while Qi1[j] % 2 == 0:
                Qi1[j] = Qi1[j]//2

    for pi in range(k,len(primes)):
        p = int(primes[pi])
        s1, s2 = tonelli(n,p)
        
        case1 = np.arange(s1,len(xi),p)
        case2 = np.arange(s2,len(xi),p)
        cases = np.unique(np.concatenate([case1,case2]))
        case1 = xi%p == s1
        case2 = xi%p == s2
        cases = np.logical_or(case1,case2)
        cases = np.where(cases == True)
        for k in np.transpose(cases):
            while (Qi1[k] % p == 0):
                Qi1[k] = Qi1[k] // p
    return Qi1
def buildMatrix(QiSmooth,A,primes):
    QiSmooth1 = np.copy(QiSmooth)
    M = np.zeros([len(QiSmooth),len(primes)+1])
    Asmooth = A[np.abs(A)==1]
    M[Asmooth < 0,0] = 1
    Asmooth = np.abs(Asmooth)
    for Q in range(0,len(QiSmooth1)):
        for pi in range(0,len(primes)):
            p = primes[pi]
            while (QiSmooth1[Q] % p == 0):
                QiSmooth1[Q] = QiSmooth1[Q] // p
                M[Q,pi+1] += 1
    return M%2

def dispMatrix(xiSmooth,QiSmooth,M,primes):
    h,w = M.shape
    disp = np.zeros([h+1,w+2])
    disp[0,0] = -3
    disp[0,1] = -2
    disp[0,2] = -1
    disp[0,3:] = primes
    disp[1:,0] = xiSmooth
    disp[1:,1] = QiSmooth
    disp[1:,2:] = M
    print(disp.astype(int))
    print(f'probability of nontrivial factor: {1-.5**(h-w)}')

##GIVEN N
n = 389*521

## DEFINE FACTOR BASE
L =np.exp(np.sqrt(np.log(n)*np.log(np.log(n))))
Bnum = (L**(np.sqrt(2)/4)).astype(int)
primes = buildFactorBase(n,Bnum - 1) #-1 is added later


## GET FIRST INTERVAL OF X
M = pow(Bnum,3)//2

nroot = int(np.sqrt(n))
xi = np.arange(nroot-M,nroot+M)
Qi = Q(xi,n)

A = sieve(n,primes,xi,Qi)
QiSmooth = Qi[np.where(np.abs(A) == 1)]
xiSmooth = xi[np.where(np.abs(A) == 1)]
print(f'Smooth x_i: {xiSmooth}')
print(f'Smooth Q(x_i): {QiSmooth}')

M = buildMatrix(QiSmooth,A,primes)
print(M)
dispMatrix(xiSmooth,QiSmooth,M,primes)
## FIND SMOOTH NUMBERS

##GIVEN N
## FIND THE LEGENDRE SYMBOL
## DETERMINE PRIME BASE
## GET FIRST INTERVAL OF X
## FIND SMOOTH NUMBERS

########################### TO DO STILL ##########################################
#remove all 0 values
## FIND LINEAR COMB = 0 (IF NOT POSSIBLE DO 2ND INTERVAL OF X)
## TEST GCD