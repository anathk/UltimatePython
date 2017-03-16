import tensorflow as tf

def feedForwardAutoencoder(saver_path, images):
  """Compute the image features, using the saved sAE model.

  Args:
    saver_path: the path to the saved sAE model.
    images: a numpy 2d array storing the raw images.

  Returns:
    A numpy 2d array storing the extracted features.
  """
  with tf.Graph().as_default():
    with tf.Session() as sess:
      # Restore trained sAE model.
      saver = tf.train.import_meta_graph(saver_path + '.meta')
      saver.restore(sess, saver_path)

      # Restore the parameters of the sAE model.
      [W1, b1, W2, b2] = tf.get_collection('sparseAE')

      ##---------------------- YOUR CODE HERE -------------------------
      #  Use the restored parameters to compute the features for each
      #  image in images.
      features = 

  return features

