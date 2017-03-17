"""Builds the MNIST softmax network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the "trainSoftmax.py" file and not meant to be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
# In this exercise, we split dataet into two parts: each ahs 5 classes
NUM_CLASSES = 5

# The MNIST images are always 28x28 pixels.
FEATURE_SIZE = 200


def inference(images):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    
  Returns:
    logits: Output tensor with the computed logits.
  """
  # Linear
  with tf.name_scope('softmax_linear'):
  ## ---------- YOUR CODE HERE --------------------------------------
    W = tf.Variable(tf.zeros([FEATURE_SIZE, NUM_CLASSES]))
    b = tf.Variable(tf.zeros([NUM_CLASSES]))
    logits = tf.matmul(images, W) + b

    tf.add_to_collection('softmax_linear', W)
    tf.add_to_collection('softmax_linear', b)
  # ------------------------------------------------------------------
  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  ## ---------- YOUR CODE HERE --------------------------------------

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels= labels)
  loss = tf.reduce_mean(cross_entropy)

  # ------------------------------------------------------------------
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  
  ## ---------- YOUR CODE HERE --------------------------------------



  # ------------------------------------------------------------------


def softmaxPredict(saver_path, features, labels):
  """Compute the correct prediction for the test features.

  Args:
    saver_path: the path of saved parameters
    features:  test features
    labels:    test labels

  Returns:
    An integer to represent number of correct prediction
  """
  with tf.Graph().as_default():
    with tf.Session() as sess:
      # Restore trained softmax model.
      saver = tf.train.import_meta_graph(saver_path + '.meta')
      saver.restore(sess, saver_path)

      # Restore the parameters of the trained softmax classifier.
      [W, b] = tf.get_collection('softmax_linear')

      ##---------------------- YOUR CODE HERE -------------------------
      # Use the restored parameters to predict the test labels.


      # ------------------------------------------------------------------
