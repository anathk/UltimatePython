import numpy as np

##======================================================================
## Implement sparseAutoencoderCost
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

# cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, decay,
#                                    sparsityParam, beta, patches)

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, decay,
                          rho, beta, data):
  """Compute cost and gradient for the Sparse AutoEncoder.

    Args:
      visibleSize: the number of input units (probably 64) 
      hiddenSize: the number of hidden units (probably 25) 
      decay: weight decay parameter
      rho: the desired average activation \rho for the hidden units
      beta: weight of sparsity penalty term
      data: The 64x10000 matrix containing the training data.
            So, data[:,i] is the i-th training example. 
  """
  
  # The input theta is a vector (because minFunc expects the parameters to be a vector). 
  # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
  # follows the notation convention of the lecture notes. 

  W1 = np.reshape(theta[: hiddenSize * visibleSize], (hiddenSize, visibleSize))
  W2 = np.reshape(theta[hiddenSize * visibleSize: 2 * hiddenSize * visibleSize], (visibleSize, hiddenSize))
  b1 = theta[2 * hiddenSize * visibleSize: 2 * hiddenSize * visibleSize + hiddenSize]
  b2 = theta[2 * hiddenSize * visibleSize + hiddenSize:]

  # Cost and gradient variables (your code needs to compute these values). 
  # Here, we initialize them to zeros. 
  cost = 0
  W1grad = np.zeros(W1.shape) 
  W2grad = np.zeros(W2.shape)
  b1grad = np.zeros(b1.shape) 
  b2grad = np.zeros(b2.shape)

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute the cost/optimization objective J(W,b) for the Sparse Autoencoder,
  #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
  #
  # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
  # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
  # as b1, etc.  Your code should set W1grad to be the partial derivative of J(W,b) with
  # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J(W,b) 
  # with respect to the input parameter W1(i,j).
  # 
  # Stated differently, if we were using batch gradient descent to optimize the parameters,
  # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
  #

  # shape(data, 1) % 64
  # shape(W1)   % 25 64
  # shape(W2)   % 64 25
  # shape(b1)   % 25  1
  # shape(b2)   % 64  1
  n, m = data.shape

  # Feed forward
  z2 = np.dot(W1, data) + np.tile(b1, (m, 1)).transpose()
  a2 = sigmoid(z2)
  z3 = np.dot(W2, a2) + np.tile(b2, (m, 1)).transpose()
  h = sigmoid(z3)

  sparse_rho = np.tile(rho, hiddenSize)
  rho_hat = np.sum(a2, axis=1) / m
  sparse_delta = np.tile(-sparse_rho / rho_hat + (1 - sparse_rho) / (1 - rho_hat), (m, 1)).transpose()

  squared_error = np.sum((h - data) ** 2) / (2 * m)
  weight_decay = (decay / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
  sparse_term = beta * np.sum(sparse_rho * np.log(sparse_rho / rho_hat) + (1 - sparse_rho) * np.log((1 - sparse_rho) / (1 - rho_hat)))


  cost = squared_error + weight_decay + sparse_term


  delta3 = -(data - h) * sigmoid(z3) * (1 - sigmoid(z3))
  delta2 = (np.dot(W2.transpose(), delta3) + beta * sparse_delta) * sigmoid(z2) * (1 - sigmoid(z2))
  W1grad = np.dot(delta2, data.transpose()) / m + decay * W1
  W2grad = np.dot(delta3, a2.transpose()) / m + decay * W2
  b1grad = np.sum(delta2, axis=1) / m
  b2grad = np.sum(delta3, axis=1) / m
  #-------------------------------------------------------------------
  # After computing the cost and gradient, we will convert the gradients back
  # to a vector format (suitable for minFunc).  Specifically, we will unroll
  # your gradient matrices into a vector.
  grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

  return cost, grad



#-------------------------------------------------------------------
# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
