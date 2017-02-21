## Sparse AutoEncoder Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the sAE exercise.
#  You will need to complete the code in sparseAutoencoder.py, 
#  sampleNaturalImages.py and sampleDigitImages.py.
#  For the purpose of completing the assignment, you do not need to
#  change the code in this file. 
#

import argparse
import sys
import math

import numpy as np
from numpy.linalg import norm
from numpy.random import randint, uniform
from scipy.optimize import fmin_l_bfgs_b

from sparseAutoencoder import sparseAutoencoderCost
from computeNumericalGradient import computeNumericalGradient
from displayNetwork import displayNetwork

def initializeParameters(hiddenSize, visibleSize):
  """Initialize parameters randomly based on layer sizes.
  """
  # We'll choose weights uniformly from the interval [-r, r].
  r  = math.sqrt(6) / math.sqrt(hiddenSize + visibleSize + 1)
  W1 = uniform(size = (hiddenSize, visibleSize)) * 2 * r - r
  W2 = uniform(size = (visibleSize, hiddenSize)) * 2 * r - r

  b1 = np.zeros(hiddenSize)
  b2 = np.zeros(visibleSize)

  # Convert weights and bias gradients to the vector form.
  # This step will "unroll" (flatten and concatenate together) all 
  # your parameters into a vector, which can then be used with minFunc. 
  theta = np.hstack((W1.ravel(), W2.ravel(), b1, b2))

  return theta


##---------------- Main program -----------------
parser = argparse.ArgumentParser('Sparse AutoEncoder Exercise.')
parser.add_argument('-t', '--input_type',
                    type=str,
                    choices = ['natural', 'digits'],
                    default='natural',
                    help = 'Type of images used for training.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../../mnist/data',
                    help='Directory to put the input MNIST data.')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Used for gradient checking.')

FLAGS, unparsed = parser.parse_known_args()


##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

if FLAGS.input_type == 'natural':
  visibleSize = 8*8   # number of input units 
  hiddenSize = 25     # number of hidden units 
  sparsityParam = 0.01  # desired average activation \rho of the hidden units.
  decay = 0.0001      # weight decay parameter       
  beta = 3            # weight of sparsity penalty term
else:
  visibleSize = 28*28
  hiddenSize = 196
  sparsityParam = 0.1
  decay = 3e-3
  beta = 3


##======================================================================
## Implement sampleImages
#
#  After implementing sampleImages, the display_network command should
#  display a random sample of 200 patches from the dataset

if FLAGS.debug:
  numPatches = 10
  hiddenSize = 2
elif FLAGS.input_type == 'natural':
  numPatches = 10000
else:
  numPatches = 20000 # need more samples, because of larger inputs.
  
if FLAGS.input_type == 'natural':
  from sampleNaturalImages import sampleNaturalImages
  patches = sampleNaturalImages('IMAGES.mat', numPatches)
else:
  from sampleDigitImages import sampleDigitImages
  patches = sampleDigitImages(FLAGS.input_data_dir, numPatches)

displayNetwork(patches[:, randint(0, patches.shape[1], 200)], 8,
               'patches-' + FLAGS.input_type + '.jpg')


#  Obtain random parameters theta
theta = initializeParameters(hiddenSize, visibleSize)

##======================================================================
## Gradient Checking
#
# Hint: If you are debugging your code, performing gradient checking on smaller models 
# and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
# units) may speed things up.


if FLAGS.debug:
  # Check your cost function and derivative calculations for the sparse autoencoder.
  cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, decay, \
                                     sparsityParam, beta, patches)
  numGrad = computeNumericalGradient(lambda x: sparseAutoencoderCost(x, visibleSize, hiddenSize, decay, sparsityParam, beta, patches), theta)

  # Use this to visually compare the gradients side by side
  print(np.stack((numGrad, grad)).T)

  # Compare numerically computed gradients with the ones obtained from backpropagation
  diff = norm(numGrad - grad) / norm(numGrad + grad)
  print(diff) # Should be small. In our implementation, these values are
              # usually less than 1e-9.
  sys.exit(1) # When you got this working, Congratulations!!!
  

##======================================================================
## After verifying that your implementation of sparseAutoencoderCost is 
# correct, You can start training your sparse autoencoder with minFunc (L-BFGS).

#  Use L-BFGS to minimize the function.
theta, _, _ = fmin_l_bfgs_b(sparseAutoencoderCost, theta,
                            args = (visibleSize, hiddenSize, decay, sparsityParam, beta, patches),
                            maxiter = 400, disp = 1)


##======================================================================
## Visualization 

# Fold W1 parameters into a matrix format.
W1 = np.reshape(theta[:hiddenSize * visibleSize], (hiddenSize, visibleSize))

# Save the visualization to a file.
displayNetwork(W1.T, file_name = 'weights-' + FLAGS.input_type + '.jpg')
