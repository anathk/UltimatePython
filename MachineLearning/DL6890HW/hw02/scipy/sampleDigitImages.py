import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import numpy as np

## ---------------------------------------------------------------
def sampleDigitImages(input_data_dir, numsamples):
  """Returns 20000 random images (28x28) from the MNIST dataset.
  """

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Fill in the variable called "patches" using data 
  #  from MNIST. 

  # Get the sets of images for training, validation, and
  # test on MNIST.
  



  ## ---------------------------------------------------------------
  # For the autoencoder to work well we need to normalize the data
  # Specifically, since the output of the network is bounded between [0,1]
  # (due to the sigmoid activation function), we have to make sure 
  # the range of pixel values is also bounded between [0,1]
  
  # samples = normalizeData(samples);

  return samples


## ---------------------------------------------------------------
def normalizeData(patches):
  """Squash data to [0.1, 0.9] since we use sigmoid as the activation
  function in the output layer
  """
  
  # Remove DC (mean of images). 
  patches = patches - np.mean(patches, axis = 0)

  # Truncate to +/-3 standard deviations and scale to -1 to 1
  pstd = 3 * np.std(patches)
  patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

  # Rescale from [-1,1] to [0.1,0.9]
  patches = (patches + 1) * 0.4 + 0.1

  return patches
