import scipy.io as sio

import numpy as np
from numpy.random import randint

## ---------------------------------------------------------------
def sampleNaturalImages(matlab_file, numpatches):
  """Returns 10000 random 8x8 patches from IMAGES.mat
  """

  images = sio.loadmat(matlab_file)['IMAGES'] # load images from disk.

  patchsize = 8  # we'll use 8x8 patches

  # Initialize patches with zeros.  Your code will fill in this matrix--one
  # column per patch, 10000 columns. 
  patches = np.zeros((patchsize ** 2, numpatches))

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Fill in the variable called "patches" using data 
  #  from 'images'.  
  #  
  #  'images' is a 3D array containing 10 images
  #  For instance, images(:,:,6) is a 512x512 array containing the 6th image,
  #  The contrast on these images looks a bit off because they have
  #  been preprocessed using using "whitening."  See the lecture notes for
  #  more details. As a second example, images(21:30,21:30,1) is an image
  #  patch corresponding to the pixels in the block (21,21) to (30,30) of
  #  Image 1.
  img_rows, img_cols, numImgs = images.shape
  for i in range(numpatches):
    ithImg = np.random.randint(0, numImgs)
    rowPos = np.random.randint(0, img_rows - patchsize)
    colPos = np.random.randint(0, img_cols - patchsize)
    patch = images[rowPos:rowPos+patchsize, colPos:colPos+patchsize, ithImg].ravel()
    patches[:, i] = patch
  ## ---------------------------------------------------------------
  # For the autoencoder to work well we need to normalize the data
  # Specifically, since the output of the network is bounded between [0,1]
  # (due to the sigmoid activation function), we have to make sure 
  # the range of pixel values is also bounded between [0,1]
  patches = normalizeData(patches)

  return patches


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
