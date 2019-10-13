# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:25:24 2019

@author: Vinay Reddy
"""
#%%
import numpy as np
i=1
""" field position"""
x=np.linspace(-34,34,69)
"""initialising the parameters of Spheres and inclined Sheets"""
santa=np.zeros((70,14851))
k=np.linspace(1,1001,11)
z0=np.linspace(2,26,6)
alpha=np.linspace(np.pi/20,np.pi/2,9)
x0=np.linspace(-2,2,5)
a=np.linspace(0.5,3,4)
""" Creating a Data set for SPheres and Inclined Sheets"""
for t1 in range(0,len(k)):
    for t2 in range(0,len(z0)):
        for t3 in range(0,len(alpha)):
            for t4 in range(0,len(x0)):
                for t5 in range(0,len(a)):
                    A1=np.square(np.subtract(x,x0[t4]+a[t5])*np.sin(alpha[t3]))
                    A2=np.square(z0[t2]+a[t5]*np.cos(alpha[t3]))
                    A3=np.square(np.subtract(x,x0[t4]-a[t5])*np.sin(alpha[t3]))
                    A4=np.square(z0[t2]-a[t5]*np.cos(alpha[t3]))
                    santa[0:69,i]=np.multiply(k[t1],np.log(np.divide(np.add(A1,A2),np.add(A3,A4))))
                    santa[-1][i]=1
                    i=i+1
for t1 in range(0,len(k)):
    for t2 in range(0,len(z0)):
        for t3 in range(0,len(alpha)):
            for t4 in range(0,len(x0)):
                A1=np.multiply(np.subtract(x,x0[t4]),np.sin(alpha[t3]))
                A2=np.add(np.square(np.subtract(x,x0[t4])),z0[t2]*z0[t2])
                santa[0:69,i]=np.multiply(k[t1],np.divide(np.subtract(A1,z0[t2]*np.cos(alpha[t3])),A2))  
                santa[-1][i]=0
                i=i+1
#%%   
