# OPTIMISATION

SP Synthetic data (syntheticdata.py)
---
   I prepared a Self Potential Synthetic data of inclined sheet and horizontal Cylinder and put them in a matrix 'santa'.
   
* 'x' is a point on principle profile. 
* 'santa' is a data sets of 14851 examples in rows with profile data in columns from 1st row to 69th row and giving a binary classification * ('1' for inclined sheet profile data and '0' for horizontal cylinder profile data) in the 70th row.
* 'k' is multiplicative factor decides maximum amplitude. 
* 'z0' is the depth of the body.
* 'alpha' is the angle of polarisation.
* 'a' is width of the sheet.
* Thus different types of inclined sheets and horizontal cylinder data are produced and classified.

Artificial Neural Network (annvinay.py)
---
An Artificial Neural Network is programmed to classify Inclined Sheets and horizontal cylinders as follows,

* outputs as matrix vector('out') and input as matrix('inp'). 
* took learning rate as 'lr'=0.01.
* 7 layers with tanh activation function and 1 layer of sigmoid activation function is used to train the model.
* wi where i=1,2,3,... are weights of ith layer.
* bi where i=1,2,3,... are biases of ith layer.
* took 50 iterations to train the model as the synthetic data is easier and faster to train.

The ANN is working properly with optimised hyperparameters. 
