import numpy as np


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
            So, data(:,i) is the i-th training example.
  """

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = np.reshape(theta[: hiddenSize * visibleSize], (hiddenSize, visibleSize))
    W2 = np.reshape(theta[hiddenSize * visibleSize: 2 * hiddenSize * visibleSize], (visibleSize, hiddenSize))
    b1 = theta[2 * hiddenSize * visibleSize: 2 * hiddenSize * visibleSize + hiddenSize].reshape((-1, 1))
    b2 = theta[2 * hiddenSize * visibleSize + hiddenSize:].reshape((-1, 1))

    # Cost and gradient variables (your code needs to compute these values).
    # Here, we initialize them to zeros.
    cost = 0
    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.zeros(b1.shape)
    b2grad = np.zeros(b2.shape)

    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
    #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
    #
    # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    # as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
    # with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
    # [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
    # of the lecture notes (and similarly for W2grad, b1grad, b2grad).
    #
    # Stated differently, if we were using batch gradient descent to optimize the parameters,
    # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
    #

    data = data.T

    z2 = np.dot(W1, data) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(W2, a2) + b2
    a3 = sigmoid(z3)

    diff = a3 - data
    num = data.shape[1]

    rho_hat = np.sum(a2, axis=1) / num

    sum_se = np.sum(diff ** 2) / (2 * num)
    regulazation = (np.sum(W1 ** 2) + np.sum(W2 ** 2)) * decay / 2
    aver_activation = beta * np.sum(KL(rho, rho_hat))

    cost = sum_se + regulazation + aver_activation

    # ----------------------------------------grade------------------------------------------------

    delta3 = diff * sigmoid_derivative(z3)
    delta2 = (np.dot(W2.T, delta3).T + beta * (- (rho / rho_hat) + (1 - rho) / (1 - rho_hat))).T * sigmoid_derivative(
        z2)

    W1grad = np.dot(delta2, data.T) / num + decay * W1
    b1grad = np.sum(delta2, axis=1) / num
    W2grad = np.dot(delta3, a2.T) / num + decay * W2
    b2grad = np.sum(delta3, axis=1) / num

    # -------------------------------------------------------------------
    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad


# -------------------------------------------------------------------
# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


def KL(rho, rho_hat):
    return rho * np.log(rho / rho_hat) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
