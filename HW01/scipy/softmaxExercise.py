## Softmax Exercise 

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  softmax exercise. You will need to write the softmax cost function and
#  the softmax prediction function in softmax.py. You will also need to write
#  code in computeNumericalGradient.py.
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than the ones mentioned above
#  (however, you may be required to do so in later exercises).

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

parser = argparse.ArgumentParser('Softmax Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../../mnist/data',
                    help='Directory to put the input MNIST data.')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Used for gradient checking.')

FLAGS, unparsed = parser.parse_known_args()

##======================================================================
## STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input. 
#  We also initialise some parameters used for tuning the model.

inputSize = 28 * 28 # Size of input vector (MNIST images are 28x28)
numClasses = 10     # Number of classes (MNIST images fall into 10 classes)

decay = 1e-4 # Weight decay parameter

##======================================================================
## STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels, 
#  the input data is the images, and 
#  the output data is the labels.
#

# Change the filenames if you've saved the files under different names
# On some platforms, the files might be saved as 
# train-images.idx3-ubyte / train-labels.idx1-ubyte

# Get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
images = data_sets.train.images.T
labels = data_sets.train.labels

# labels.flags.writeable = True
# labels[labels == 0] = 10 # Remap 0 to 10

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create synthetic dataset using random data for testing

if FLAGS.debug:
  inputSize = 8
  images = randn(8, 100)
  labels = randint(0, 10, 100, dtype = np.uint8)

# Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize)

##======================================================================
## STEP 2: Implement softmaxCost
#
#  Implement softmaxCost in softmax.py. 

cost, grad = softmaxCost(theta, numClasses, inputSize, decay, images, labels)

##======================================================================
## STEP 3: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
#

if FLAGS.debug:
  # First, lets make sure your numerical gradient computation is correct for a
  # simple function.  After you have implemented computeNumericalGradient.py,
  # run the following: 
  checkNumericalGradient()

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

  # When your gradients are correct, congratulations!
                                    
##======================================================================
## STEP 4: Learning parameters
#
#  Once you have verified that your gradients are correct, 
#  you can start training your softmax regression code using L-BFGS.

theta, _, _ = fmin_l_bfgs_b(softmaxCost, theta,
                            args = (numClasses, inputSize, decay, images, labels),
                            maxiter = 100, disp = 1)
# Fold parameters into a matrix format.
theta = np.reshape(theta, (numClasses, inputSize));
    
# Although we only use 100 iterations here to train a classifier for the 
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

##======================================================================
## STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write softmaxPredict
#  (in softmaxPredict.m), which should return predictions
#  given a softmax model and the input data.

images = data_sets.test.images.T
labels = data_sets.test.labels

# labels.flags.writeable = True
# labels[labels == 0] = 10 # Remap 0 to 10

# You will have to implement softmaxPredict in softmax.py.
pred = softmaxPredict(theta, images)

acc = np.mean(labels == pred)
print('Accuracy: %0.3f%%.' % acc * 100)

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.500%
#
# If your values are too low (accuracy less than 0.91), you should check 
# your code for errors, and make sure you are training on the 
# entire data set of 60000 28x28 training images 
# (unless you modified the loading code, this should be the case)
