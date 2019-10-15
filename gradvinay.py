# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:46:56 2019

@author: Vinay Reddy
"""
#%%
import numpy as np
learningrate=0.001
x=np.asmatrix(np.linspace(-34,34,69))
L=np.subtract(np.multiply(np.subtract(x,1),np.sin(np.pi/7)),np.multiply(18,np.cos(np.pi/7)))
data = np.transpose(np.multiply(120,np.divide(L,np.add(np.square(np.subtract(x,1)),18**2))))
M=119.99995139785717
K=0.4607158569739635
x0=1.7559713848807337
z0=18.677372743366423
for i in range(0,100):
    L1=np.transpose(x)-np.multiply(x0,np.ones((69,1)))
    L1=np.subtract(np.multiply(L1,K),np.multiply(z0,np.sqrt(np.subtract(1,np.square(K)))))
    L2=np.add(np.square(np.transpose(x)-np.multiply(x0,np.ones((69,1)))),np.square(z0))
    F = np.multiply(M,np.divide(L1,L2))
    F1 = np.sum(np.square(data-F),axis=0)
    det=np.sqrt(np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),3))
    det2=np.sqrt(np.power(np.add(np.square(np.subtract(x,x0)),z0*z0),5))
    det3=np.sqrt(1-K*K)
    dK1 = np.divide(np.subtract(x,x0),det)
    dK2 = np.divide(z0*K/(det3),det)
    dK3 = np.multiply(dK1+dK2,M)
    dK=np.dot(dK3,data-F)/69
    dM=np.divide(F,M)
    dM=np.dot(np.reshape(dM,(1,69)),data-F)/69
    dx01=np.divide(np.multiply(-3*z0*(det3),np.subtract(x,x0)),det2)
    dx02=np.divide(np.multiply(3*K,np.square(np.subtract(x,x0))),det2)
    dx03=np.divide(-K,det)
    dx0=np.multiply(M,dx01+dx02+dx03)
    dx0=np.dot(dx0,data-F)/69
    dz01=np.divide(np.multiply(-3*z0*K,np.subtract(x,x0)),det2)
    dz02=np.divide(3*z0*z0*(det3),det2)
    dz03=np.divide(-(det3),det)
    dz0=np.multiply(M,dz01+dz02+dz03)
    dz0=np.dot(dz0,data-F)/69
    z0=z0-dz0*learningrate*1
    x0=x0-dx0*learningrate*100
    M=M-dM*learningrate*0.1
    K=K-dK*learningrate*0.1
