#  This file contains code that trains the softmax model. You will need to write
#  the softmax cost function and the softmax prediction function in softmax.py.
#  You will also need to write code in computeNumericalGradient.py.
  
import argparse
import sys
  
import numpy as np
from numpy.random import randn, randint
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
  
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
  
from softmax import softmaxCost, softmaxPredict
from computeNumericalGradient import computeNumericalGradient
  
  
def run_training(FLAGS, images, labels):
  # For debugging purposes, you may wish to reduce the size of the input data
  # in order to speed up gradient checking. 
  # Here, we create synthetic dataset using random data for testing
  
  if FLAGS.debug:
    inputSize = 8
    images = randn(8, 100)
    labels = randint(0, 10, 100, dtype = np.uint8)
  else:
    inputSize = FLAGS.visibleSize

  numClasses = 5
  decay = FLAGS.decay
  
  # Randomly initialise theta
  theta = 0.005 * randn(numClasses * inputSize)
  
  # Implement softmaxCost in softmax.py.   
  cost, grad = softmaxCost(theta, numClasses, inputSize, decay, images, labels)
  
  #  As with any learning algorithm, you should always check that your
  #  gradients are correct before learning the parameters.
  if FLAGS.debug:
    # First, lets make sure your numerical gradient computation is correct for a
    # simple function.  After you have implemented computeNumericalGradient.py,
    # run the following: 
    #checkNumericalGradient()
  
    numGrad = computeNumericalGradient(lambda x: softmaxCost(x, numClasses, inputSize, decay, images, labels),
                                       theta)
  
    # Use this to visually compare the gradients side by side.
    print(np.stack((numGrad, grad)).T)
  
    # Compare numerically computed gradients with those computed analytically.
    diff = norm(numGrad - grad) / norm(numGrad + grad)
    print(diff)
    sys.exit(1)
    # The difference should be small. 
    # In our implementation, these values are usually less than 1e-7.
                                      
  #  Once you have verified that your gradients are correct, 
  #  you can start training your softmax regression code using L-BFGS.
  theta, _, _ = fmin_l_bfgs_b(softmaxCost, theta,
                              args = (numClasses, inputSize, decay, images, labels),
                              maxiter = 400, disp = 1)

  # Fold parameters into a matrix format.
  theta = np.reshape(theta, (numClasses, inputSize));

  return theta
  
