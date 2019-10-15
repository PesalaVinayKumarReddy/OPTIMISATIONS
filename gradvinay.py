# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:46:56 2019

@author: Vinay Reddy
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
learningrate=0.01
x=np.asmatrix(np.linspace(-34,34,69))
L=np.subtract(np.multiply(np.subtract(x,1),np.sin(np.pi/7)),np.multiply(18,np.cos(np.pi/7)))
data = np.transpose(np.multiply(120,np.divide(L,np.add(np.square(np.subtract(x,1)),18**2))))
M=119.99995139785717
K=0.4607158569739635
x0=1.7559713848807337
z0=18.677372743366423
for i in range(0,1):
    L1=np.transpose(x)-np.multiply(x0,np.ones((69,1)))
    L1=np.subtract(np.multiply(L1,K),np.multiply(z0,np.sqrt(np.subtract(1,np.square(K)))))
    L2=np.add(np.square(np.transpose(x)-np.multiply(x0,np.ones((69,1)))),np.square(z0))
    F = np.multiply(M,np.divide(L1,L2))
    F1 = np.sum(np.square(data-F),axis=0)
    dK1 = np.divide(np.subtract(x,x0),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),1.5))
    dK2 = np.divide(z0*K/((1-K*K)**0.5),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),1.5))
    dK3 = np.multiply(dK1+dK2,M)
    dK=np.dot(dK3,data-F)/69
    dM=np.sum(np.divide(F,M),axis=0)
    dx01=np.divide(np.multiply(3*z0*((1-K*K)**0.5),np.subtract(x,x0)),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),2.5))
    dx02=np.divide(np.multiply(3*K,np.square(np.subtract(x,x0))),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),2.5))
    dx03=np.divide(-K,np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),1.5))
    dx0=np.multiply(M,dx01+dx02+dx03)
    dz01=np.divide(np.multiply(-3*z0*K,np.subtract(x,x0)),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),2.5))
    dz02=np.divide(np.multiply(-3*z0*z0*((1-K*K)**0.5),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),2.5))
    dz03=np.divide(-((1-K*K)**0.5),np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),1.5))
    x0=x0-dx0*learningrate
    M=M-dM*learningrate
    K=K-dK*learningrate