import scipy.io as sio
from scipy.misc import imsave

import numpy as np
from numpy.random import randint

import matplotlib.pyplot as plt

from displayNetwork import displayNetwork

# Function sampleIMAGESRAW().
# Returns 10000 "raw" unwhitened  patches.
def sampleIMAGESRAW(matlab_file):
  images = sio.loadmat(matlab_file)['IMAGESr'] # load images from disk.

  patchSize = 12
  numPatches = 10000

  # Initialize patches with zeros. Your code will fill in this matrix,
  # one column per patch, 10000 columns. 
  patches = np.zeros((patchSize * patchSize, numPatches))

  p = 0
  for im in range(images.shape[2]):
    # Sample Patches
    numSamples = numPatches // images.shape[2]
    for s in range(numSamples):
      y = randint(images.shape[0] - patchSize)
      x = randint(images.shape[1] - patchSize)
      sample = images[y : y + patchSize, x : x + patchSize, im]
      patches[:, p] = sample.ravel()
      p = p + 1

  return patches


##================================================================
## Step 0a: Load data
#  Here we provide the code to load natural image data into x.
#  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
#  the raw image data from the kth 12x12 image patch sampled.
#  You do not need to change the code below.

x = sampleIMAGESRAW('IMAGES_RAW.mat')
randsel = randint(x.shape[1], size=196) # A random selection of samples for visualization.
displayNetwork(x[:, randsel], file_name = 'figure7.jpg', opt_normalize = True)

##================================================================
## Step 0b: Zero-mean the data (by row)

# -------------------- YOUR CODE HERE -------------------- 

x0 = ?

# -------------------------------------------------------- 

##================================================================
## Step 1a: Implement PCA to obtain xRot
#  Implement PCA to obtain xRot, the matrix in which the data is expressed
#  with respect to the eigenbasis of sigma, which is the matrix U.


# -------------------- YOUR CODE HERE -------------------- 

xRot = ?

# -------------------------------------------------------- 

##================================================================
## Step 1b: Check your implementation of PCA
#  The covariance matrix for the data expressed with respect to the basis U
#  should be a diagonal matrix with non-zero entries only along the main
#  diagonal. We will verify this here.
#  Write code to compute the covariance matrix, 'covar'. 
#  When visualised as an image, you should see a straight line across the
#  diagonal (non-zero entries) against a blue background (zero entries).

# -------------------- YOUR CODE HERE -------------------- 

covar = ?

# -------------------------------------------------------- 

# Visualise the covariance matrix. You should see a line across the
# diagonal against a blue background.
imsave('figure8.jpg', covar)

##================================================================
## Step 2: Find k, the number of components to retain
#  Write code to determine k, the number of components to retain in order
#  to retain at least 99% of the variance.

# -------------------- YOUR CODE HERE -------------------- 

k = ?

# -------------------------------------------------------- 

##================================================================
## Step 3: Implement PCA with dimension reduction
#  Now that you have found k, you can reduce the dimension of the data by
#  discarding the remaining dimensions. In this way, you can represent the
#  data in k dimensions instead of the original 144, which will save you
#  computational time when running learning algorithms on the reduced
#  representation.
# 
#  Following the dimension reduction, invert the PCA transformation to produce 
#  the matrix xHat, the dimension-reduced data with respect to the original basis.
#  Visualise the data and compare it to the raw data. You will observe that
#  there is little loss due to throwing away the principal components that
#  correspond to dimensions with low variation.

# -------------------- YOUR CODE HERE -------------------- 

xHat = ?

# -------------------------------------------------------- 

# Visualise the data, and compare it to the raw data
# You should observe that the raw and processed data are of comparable quality.
# For comparison, you may wish to generate a PCA reduced image which
# retains only 90% of the variance.

displayNetwork(xHat[:, randsel], file_name = 'figure9.jpg', opt_normalize = True)

##================================================================
## Step 4a: Implement PCA with whitening and regularisation
#  Implement PCA with whitening and regularisation to produce the matrix
#  xPCAWhite. 

epsilon = 0.1

# -------------------- YOUR CODE HERE -------------------- 

xPCAWhite = ?

# -------------------------------------------------------- 

##================================================================
## Step 4b: Check your implementation of PCA whitening 
#  Check your implementation of PCA whitening with and without regularisation. 
#  PCA whitening without regularisation results a covariance matrix 
#  that is equal to the identity matrix. PCA whitening with regularisation
#  results in a covariance matrix with diagonal entries starting close to 
#  1 and gradually becoming smaller. We will verify these properties here.
#  Write code to compute the covariance matrix, covar. 
#
#  Without regularisation (set epsilon to 0 or close to 0), 
#  when visualised as an image, you should see a red line across the
#  diagonal (one entries) against a blue background (zero entries).
#  With regularisation, you should see a red line that slowly turns
#  blue across the diagonal, corresponding to the one entries slowly
#  becoming smaller.

# -------------------- YOUR CODE HERE -------------------- 

covar = ?

# -------------------------------------------------------- 

# Visualise the covariance matrix. You should see a red line across the
# diagonal against a blue background.
plt.figure(10)
plt.title('PCAWhite Covariance matrix')
plt.imshow(covar)
imsave('figure10.jpg', covar)

##================================================================
## Step 5: Implement ZCA whitening
#  Now implement ZCA whitening to produce the matrix xZCAWhite. 
#  Visualise the data and compare it to the raw data. You should observe
#  that whitening results in, among other things, enhanced edges.

# -------------------- YOUR CODE HERE -------------------- 

xZCAWhite = ?

# -------------------------------------------------------- 

# Visualise the data, and compare it to the raw data.
# You should observe that the whitened images have enhanced edges.
displayNetwork(xZCAWhite[:, randsel], file_name = 'figure11.jpg', opt_normalize = True)
