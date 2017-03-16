#  This file contains code that trains the sAE.
#  You will need to complete the code in sparseAutoencoder.py. 
  
import argparse
import sys
import math
import pickle
  
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
  
  
##---------------- train sparse autoencoder -----------------
def run_training(FLAGS, patches):
  ##======================================================================
  ## STEP 1: Here we provide the relevant parameters values that will
  #  allow your sparse autoencoder to get good filters; you do not need to 
  #  change the parameters below.
  
  visibleSize = FLAGS.visibleSize  # number of input units 
  hiddenSize = FLAGS.hiddenSize    # number of hidden units 
  sparsityParam = FLAGS.rho        # desired average activation \rho of the hidden units.
  decay = FLAGS.decay              # weight decay parameter       
  beta = FLAGS.beta                # weight of sparsity penalty term
  
  #  Obtain random parameters theta
  theta = initializeParameters(hiddenSize, visibleSize)
  
  ##======================================================================
  ## STEP 2: Implement sparseAutoencoderCost
  #
  #  You can implement all of the components (squared error cost, weight decay term,
  #  sparsity penalty) in the cost function at once, but it may be easier to do 
  #  it step-by-step and run gradient checking (see STEP 3) after each step.  We 
  #  suggest implementing the sparseAutoencoderCost function using the following steps:
  #
  #  (a) Implement forward propagation in your neural network, and implement the 
  #      squared error term of the cost function.  Implement backpropagation to 
  #      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
  #      to verify that the calculations corresponding to the squared error cost 
  #      term are correct.
  #
  #  (b) Add in the weight decay term (in both the cost function and the derivative
  #      calculations), then re-run Gradient Checking to verify correctness. 
  #
  #  (c) Add in the sparsity penalty term, then re-run Gradient Checking to 
  #      verify correctness.
  #
  #  Feel free to change the training settings when debugging your
  #  code.  (For example, reducing the training set size or 
  #  number of hidden units may make your code run faster; and setting beta 
  #  and/or lambda to zero may be helpful for debugging.)  However, in your 
  #  final submission of the visualized weights, please use parameters we 
  #  gave in Step 0 above.
  
  cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, decay,
                                     sparsityParam, beta, patches)
  
  ##======================================================================
  ## STEP 3: Gradient Checking
  #
  # Hint: If you are debugging your code, performing gradient checking on smaller models 
  # and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
  # units) may speed things up.
  
  
  if FLAGS.debug:
    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
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
  ## STEP 4: After verifying that your implementation of
  #  sparseAutoencoderCost is correct, You can start training your sparse
  #  autoencoder with minFunc (L-BFGS).
  
  #  Randomly initialize the parameters.
  theta = initializeParameters(hiddenSize, visibleSize)
  
  #  Use L-BFGS to minimize the function.
  theta, _, _ = fmin_l_bfgs_b(sparseAutoencoderCost, theta,
                              args = (visibleSize, hiddenSize, decay, sparsityParam, beta, patches),
                              maxiter = 400, disp = 1)

  # save the learned parameters to external file
  pickle.dump(theta, open(FLAGS.log_dir + '/' + FLAGS.params_file, 'wb'))
  
  ##======================================================================
  ## STEP 5: Visualization 
  
  # Fold W1 parameters into a matrix format.
  W1 = np.reshape(theta[:hiddenSize * visibleSize], (hiddenSize, visibleSize))
  
  # Save the visualization to a file.
  displayNetwork(W1.T, file_name = 'weights_digits.jpg')

  return theta


