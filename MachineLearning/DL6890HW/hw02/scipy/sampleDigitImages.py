import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from os import listdir
from os import path
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
  # Image data format:
  # [offset] [type]          [value]          [description]
  # 0000     32 bit integer  0x00000803(2051) magic number
  # 0004     32 bit integer  60000            number of images
  # 0008     32 bit integer  28               number of rows
  # 0012     32 bit integer  28               number of columns
  # 0016     unsigned byte   ??               pixel
  # 0017     unsigned byte   ??               pixel
  # ........
  # xxxx     unsigned byte   ??               pixel
  # Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

  files = [mnistfile for mnistfile in listdir(input_data_dir) if "images" in mnistfile]
  samples = None
  for imgfile in files:
    with open(path.join(input_data_dir, imgfile), "r") as f:
      magic = np.fromfile(f, dtype=np.dtype(">i4"), count=1)
      numImags = np.fromfile(f, dtype=np.dtype(">i4"), count=1)
      rows = np.fromfile(f, dtype=np.dtype(">i4"), count=1)
      cols = np.fromfile(f, dtype=np.dtype(">i4"), count=1)
      images = np.fromfile(f, dtype=np.ubyte)
      images = images.reshape((numImags, rows * cols))
      if samples is None:
        samples = images.copy()
      else:
        samples = np.concatenate((samples, images), axis=0)

  samples = samples[np.random.randint(0, samples.shape[0], numsamples)]


  ## ---------------------------------------------------------------
  # For the autoencoder to work well we need to normalize the data
  # Specifically, since the output of the network is bounded between [0,1]
  # (due to the sigmoid activation function), we have to make sure 
  # the range of pixel values is also bounded between [0,1]
  
  samples = normalizeData(samples);

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

# if __name__ == "__main__":
#   print(sampleDigitImages("../../mnist/data", 100).shape)