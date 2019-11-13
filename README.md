# OPTIMISATION TECHNIQUE to find the Parameters of the unknown Geometry from the principle profile data

Auther:  "Pesala Vinay Kumar Reddy"

licensed under [MIT License](LICENSE)

Files
---
[syntheticdata](syntheticdata.py)

[ANN](annvinay.py)

[Particle Swarm Optimisation](psovinay.py)

SP Synthetic data (syntheticdata.py)
---
   I prepared a Self Potential Synthetic data of inclined sheet and horizontal Cylinder and put them in a matrix 'santa'. santa is a training data set for ANN.
   
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

Particle Swarm Optimisation (psovinay.py)
---

* After classifying the data from the Neural Network to find the geometry of the body in the subsurface the earth, we do Particle Swarm Optimisation.
* PSO is global optimisation technique, so we get the best fit parameters.
* It is started with 100 particles.
* We define a random initialisation of all parameters in the knnown range and find the synthetic data of that particular geometry found from ANN.
* Then from the use of Particle Swarm optimisation Algorithm we find the global best solution attained by all the particles which were randomly initialised.
* All the particles posses different velocities at each iteration. These velocities are defined by global best found at the moment and the individual particle best found at the moment.
* Therefore in the end Some of the particles end up near to the global best parameters and a few particles end up in local best parameters fit.
* Thus we find global best parameters of the data provided.

# Short comings

* Did not apply for Real Field data.
* ANN program is written to classify for only two diffwrent types of models Where it can be written for spheres, vertical cylinders as well
* Inputs in ANN is not scaled.
* Hyper Parameters of ANN are optimised manually.
* Particle Swarm Optimisation Program is written only for Horizontal Cylinder.
