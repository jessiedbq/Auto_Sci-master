# Auto_Sci  
This repository contains code for the final project of Machine Learning @ NYU Spring 2019.  
Bingqian Deng  
bd1397@nyu.edu  


## Paper  
The project aims to implement the algorithm FGPGM in the paper "Fast Gaussian Process Based Gradient Matching for Parameter Identification in Systems of Nonlinear ODEs" by Philippe Wenk, Alkis Gotovos, Stefan Bauer, Nico Gorbach, Andreas Krause and Joachim M. Buhmann. (http://arxiv.org/abs/1804.04378).  

## Currently working Code  

Due to the difficulty of the project: system simulation + GP regression for time series + MCMC for Approximate Inference(consulted the author, he spent around 3 months to implement). Only GP regression for time series is finished.
To run the GP regression for FHN system, just run the GP_FHN jupyter notebook in the the folder "examples"  

The code provided is written in Python 3.7.1, and relies on the following libraries:  
pytorch-nightly           1.0.0.dev20180929  
matplotlib                3.0.2  
numpy                     1.15.4  
scipy                     1.1.0  
pandas                    0.23.4
