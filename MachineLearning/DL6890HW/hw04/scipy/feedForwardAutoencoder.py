import pickle
import numpy as np


def feedForwardAutoencoder(FLAGS, images):
  # Load parameters from file.
  theta = pickle.load(open(FLAGS.log_dir + '/' + FLAGS.params_file, 'rb'))


  ## ---------- YOUR CODE HERE --------------------------------------
  # Step1.
  #   Recover W1 and b1 from theta.
  W1 =
  b1 = 

  # Step2.
  #   Forward propagation with W1 and b1, to compute the sAE features.

  features = 
  #-------------------------------------------------------------------

  return features


def sigmoid(x):
  return 1 / (1 + np.exp(-x))
