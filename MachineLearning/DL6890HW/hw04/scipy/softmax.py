import numpy as np
from scipy.sparse import coo_matrix

def softmaxCost(theta, numClasses, inputSize, decay, data, labels):
  """Computes and returns the (cost, gradient)

  numClasses - the number of classes 
  inputSize - the size N of the input vector
  lambda - weight decay parameter
  data - the N x M input matrix, where each row data[i, :] corresponds to
         a single sample
  labels - an M x 1 matrix containing the labels corresponding for the input data
  """

  # Unroll the parameters from theta
  theta = np.reshape(theta, (numClasses, inputSize))

  # get the number of data
  numCases = data.shape[1]

  groundTruth = coo_matrix((np.ones(numCases, dtype = np.uint8),
                            (labels, np.arange(numCases)))).toarray()
  cost = 0;
  thetagrad = np.zeros((numClasses, inputSize))

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute the cost and gradient for softmax regression.
  #                You need to compute thetagrad and cost.
  #                The groundTruth matrix might come in handy.


  # ------------------------------------------------------------------
  # Unroll the gradient matrices into a vector for the optimization function.
  grad = thetagrad.ravel()

  return cost, grad


def softmaxPredict(theta, data):
  """Computes and returns the softmax predictions in the input data.

  theta - model parameters trained using softmaxTrain,
          a numClasses x inputSize matrix.
  data - the M x N input matrix, where each row data[i,:] corresponds to
         a single sample.
  """

  #  Your code should produce the prediction matrix pred,
  #  where pred(i) is argmax_c P(c | x(i)).
 
  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute pred using theta.

  # ---------------------------------------------------------------------

  return pred
