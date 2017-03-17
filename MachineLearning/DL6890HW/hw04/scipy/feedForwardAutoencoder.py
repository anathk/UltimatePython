import pickle
import numpy as np


def feedForwardAutoencoder(FLAGS, images):
  # Load parameters from file.
  theta = pickle.load(open(FLAGS.log_dir + '/' + FLAGS.params_file, 'rb'))


  ## ---------- YOUR CODE HERE --------------------------------------
  # Step1.
  #   Recover W1 and b1 from theta.
  hidden_size = FLAGS.hiddenSize
  visible_size = FLAGS.visibleSize
  W1 = np.reshape(theta[: hidden_size * visible_size], (hidden_size, visible_size))
  #W1 = theta[0: hidden_size * visible_size].reshape((hidden_size, visible_size))
  b1 = theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size + hidden_size]

  # Step2.
  #   Forward propagation with W1 and b1, to compute the sAE features.
  features = sigmoid(((np.dot(W1, images.T)).T + b1).T)

  #-------------------------------------------------------------------

  return features


def sigmoid(x):
  return 1 / (1 + np.exp(-x))
