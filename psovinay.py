# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:43:48 2019

@author: Devendra Mishra
"""
#%%
""" 
After classifying the data from the Neural Network to find the geometry of the body in the subsurface the earth, we do Particle Swarm Optimisation
PSO is global optimisation technique, so we get the best fit parameters
It is started with 100 particles
"""
import numpy as np
import math
import matplotlib.pyplot as plt
"""
M is multiplicative factor decides ampllitude is randomised between 0 and 150
K is 'sin(alpha)' randomised in between 0 and 1
x0 is the location at which the body is present on the principle profile randomised between -3 and 3
z0 is the depth of the body randomised between 1 and 30
alpha is the angle of polarisation
Here 'data' matrix is created synthetic 
"""
K = np.random.rand(1,100)
M = np.random.rand(1,100)*150;
x0 = np.asmatrix(np.linspace(-3,3,100))
z0 = np.asmatrix(np.linspace(1,30,100))
X = np.concatenate((M,K,x0,z0),axis=0)
x=np.asmatrix(np.linspace(-34,34,69))
L=np.subtract(np.multiply(np.subtract(x,1),np.sin(np.pi/7)),np.multiply(18,np.cos(np.pi/7)))
data = np.transpose(np.multiply(120,np.divide(L,np.add(np.square(np.subtract(x,1)),18**2))))
F = np.zeros((69,100))
"""
c1 and c2 are the congnigent and social multiplication factors of the velocities
"""
c1 = np.array([[1.],[0.01],[0.2],[0.1]])
c2 = np.array([[1],[0.02],[0.3],[0.2]])
mvaluebest=100000
"""" 
F1best is misfit of the best fit
"""
F1best=np.multiply(np.ones((1,100)),10000000)
"""
initialising the individual particle best
"""
p=np.zeros((4,100))
""" 
Random initialisation of velocities
"""
V=np.random.rand(4,100)*0.01
cost=np.zeros((99))
for it in range(1,100):
    M=X[:1]
    K=np.asmatrix(X[1])
    x0=np.asmatrix(X[2])
    z0=X[-1:]
    L1=np.subtract(np.transpose(x),np.multiply(x0,np.ones((69,1))))
    L1=np.subtract(np.multiply(L1,K),np.multiply(z0,np.sqrt(np.subtract(1,np.square(K)))))
    L2=np.add(np.square(np.subtract(np.transpose(x),np.multiply(x0,np.ones((69,1))))),np.square(z0))
    """ 
    F is the misfit of the individual particles of at their best fit throughout the iterations
    """
    F = np.multiply(M,np.divide(L1,L2))
    """ 
    F1best is the misfit of the best fit found during the iterations
    """
    F1 = np.sum(np.square(np.subtract(data,F)),axis=0)
    mi = np.argmin(F1)
    F1=np.array(F1)
    mvalue=F1[0][mi]
    """
    Updating the Global best particle when it meets the condition that the program finds the lesser misfit value than previously found
    """
    if mvalue<mvaluebest:
        pg = np.multiply(np.ones((4,100)),np.asmatrix(X[:,[mi]]))
        mvaluebest = mvalue
    logic = F1<F1best
    p = np.add(np.multiply(logic,X),np.multiply(np.subtract(1,logic),p))
    r=np.random.rand(1,2)
    """
    Updating the velocities at each iteration
    """
    V = V+np.multiply(r[0][0],np.multiply(c1,(p-X)))+np.multiply(r[0][1],np.multiply(c2,(pg-X)))
    X = X+V
    cost[it-1] = mvaluebest
m11=np.array(pg[0,[0]])[0][0]
k11=np.array(pg[1,[0]])[0][0]
k11=math.asin(k11)
x00=np.array(pg[2,[0]])[0][0]
z00=np.array(pg[3,[0]])[0][0]
L=np.subtract(np.multiply(np.subtract(x,x00),np.sin(k11)),np.multiply(z00,np.cos(k11)))
result = np.transpose(np.multiply(m11,np.divide(L,np.add(np.square(np.subtract(x,x00)),z00**2))))
plt.plot(x.T, data, label ="Data")
plt.plot(x.T, result, label = "result")
plt.xlabel('Principle profile')
plt.ylabel('SP(mV)')
plt.title('plot of SP field data and computed data')
plt.legend()
plt.show()
plt.plot(cost)
plt.xlabel('iterations')
plt.ylabel('misfit')
plt.title('cost function vs Iterations')
plt.show()
print("multiplicative factor = ",m11)
print("angle of polarisation in radians = ",k11)
print("position of the sphere= ",x00)
print("depth of the sphere= ",z00)
    
