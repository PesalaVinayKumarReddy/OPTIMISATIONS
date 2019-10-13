# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:51:16 2019

@author: Vinay Reddy
"""
#%%
import numpy as np
"""out is a row vector of true and false deciding sphere or cylinder"""
out=santa[-1:]
"""inp is Synthetic SP data of corresponding Spheres and Cylinders"""
inp=santa[:-1]
lr = 0.01
""" Initialising Weights and Biases"""
w1 = np.random.rand(69,69)*0.1
b1 = np.zeros((69,1))
w2 = np.random.rand(69,69)*0.1
b2 = b1
w3 = np.random.rand(69,69)*0.1
b3 = b1
w4 = np.random.rand(30,69)*0.1
b4 = np.zeros((30,1))
w5 = np.random.rand(30,30)*0.1
b5 = np.zeros((30,1))
w6 = np.random.rand(20,30)*0.1
b6 = np.zeros((20,1))
w7 = np.random.rand(10,20)*0.1
b7 = np.zeros((10,1))
w8 = np.random.rand(1,10)*0.1
b8=np.zeros((1,1))
for iterations in range(0,50):
    """ Forward model"""
    x1 = np.add(np.dot(w1,inp),b1)
    z1 = np.tanh(x1)
    x2 = np.add(np.dot(w2,z1),b2)
    z2 = np.tanh(x2)
    x3 = np.add(np.dot(w3,z2),b3)
    z3 = np.tanh(x3)
    x4 = np.add(np.dot(w4,z3),b4)
    z4 = np.tanh(x4)
    x5 = np.add(np.dot(w5,z4),b5)
    z5 = np.tanh(x5)
    x6 = np.add(np.dot(w6,z5),b6)
    z6 = np.tanh(x6)
    x7 = np.add(np.dot(w7,z6),b7)
    z7 = np.tanh(x7)
    x8 = np.add(np.dot(w8,z7),b8)
    z8 = np.divide(1,np.add(1,np.exp(-x8)))
    """ Back Propagation"""
    dx8 = np.subtract(z8,out)
    dw8 = np.dot(dx8,np.transpose(z7))
    db8 = np.asmatrix(np.sum(dx8,axis=1))
    dz7 = np.dot(np.transpose(w8),dx8)
    dx7 = np.multiply(dz7,np.square(np.divide(1,np.cosh(x7))))
    dw7 = np.dot(dx7,np.transpose(z6))
    db7 = np.reshape(np.asmatrix(np.sum(dx7,axis=1)),(10,1))
    dz6 = np.dot(np.transpose(w7),dx7)
    dx6 = np.multiply(dz6,np.square(np.divide(1,np.cosh(x6))))
    dw6 = np.dot(dx6,np.transpose(z5))
    db6 = np.reshape(np.asmatrix(np.sum(dx6,axis=1)),(20,1))
    dz5 = np.dot(np.transpose(w6),dx6)
    dx5 = np.multiply(dz5,np.square(np.divide(1,np.cosh(x5))))
    dw5 = np.dot(dx5,np.transpose(z4))
    db5 = np.reshape(np.asmatrix(np.sum(dx5,axis=1)),(30,1))
    dz4 = np.dot(np.transpose(w5),dx5)
    dx4 = np.multiply(dz4,np.square(np.divide(1,np.cosh(x4))))
    dw4 = np.dot(dx4,np.transpose(z3))
    db4 = np.reshape(np.asmatrix(np.sum(dx4,axis=1)),(30,1))
    dz3 = np.dot(np.transpose(w4),dx4)
    dx3 = np.multiply(dz3,np.square(np.divide(1,np.cosh(x3))))
    dw3 = np.dot(dx3,np.transpose(z2))
    db3 = np.reshape(np.asmatrix(np.sum(dx3,axis=1)),(69,1))
    dz2 = np.dot(np.transpose(w3),dx3)
    dx2 = np.multiply(dz2,np.square(np.divide(1,np.cosh(x2))))
    dw2 = np.dot(dx2,np.transpose(z1))
    db2 = np.reshape(np.asmatrix(np.sum(dx2,axis=1)),(69,1))
    dz1 = np.dot(np.transpose(w2),dx2)
    dx1 = np.multiply(dz1,np.square(np.divide(1,np.cosh(x1))))
    dw1 = np.dot(dx1,np.transpose(inp))
    db1 = np.reshape(np.asmatrix(np.sum(dx1,axis=1)),(69,1))
    """ Updating the Parameters (weights and biases) """
    w8 = np.subtract(w8,np.multiply(lr,dw8))
    b8 = np.subtract(b8,np.multiply(lr,db8))
    w7 = np.subtract(w7,np.multiply(lr,dw7))
    b7 = np.subtract(b7,np.multiply(lr,db7))
    w6 = np.subtract(w6,np.multiply(lr,dw6))
    b6 = np.subtract(b6,np.multiply(lr,db6))
    w5 = np.subtract(w5,np.multiply(lr,dw5))
    b5 = np.subtract(b5,np.multiply(lr,db5))
    w4 = np.subtract(w4,np.multiply(lr,dw4))
    b4 = np.subtract(b4,np.multiply(lr,db4))
    w3 = np.subtract(w3,np.multiply(lr,dw3))
    b3 = np.subtract(b3,np.multiply(lr,db3))
    w2 = np.subtract(w2,np.multiply(lr,dw2))
    b2 = np.subtract(b2,np.multiply(lr,db2))
    w1 = np.subtract(w1,np.multiply(lr,dw1))
    b1 = np.subtract(b1,np.multiply(lr,db1))
