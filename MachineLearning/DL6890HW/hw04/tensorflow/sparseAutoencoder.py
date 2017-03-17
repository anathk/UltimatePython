"""Builds the sAE network and operations to compute the cost.

Implements the loss/training pattern for feature learning.

1. los() - Builds the model as far as is required for running the network
forward to make predictions. Adds to the inference model the layers required
to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the "trainAE.py" file and not meant to be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

def loss(images, visible_size, hidden_size, decay, rho, beta):
  """Build the sparseAE model up to where it may be used for inference
  and add operations to compute the loss.

  Args:
    images: Images placeholder, from inputs().
    hidden_size: number of filters.
    decay: L2 regularization hyper-parameter.
    rho: sparsity level hyper-parameter.
    beta: sparsity hyper-parameter.
    
  Returns:
    cost: The computed cost.
  """
  # We'll choose weights uniformly from the interval [-r, r].
  r  = math.sqrt(6) / math.sqrt(hidden_size + visible_size + 1)

  with tf.name_scope('sparseAE'):
    ## ---------- YOUR CODE HERE --------------------------------------

    # Inference Variables
    weights1 = tf.Variable(tf.random_uniform([hidden_size, visible_size], -r, r), name='weights1')
    biases1 = tf.Variable(tf.zeros([hidden_size, 1]), name="biases1")
    weights2 = tf.Variable(tf.random_uniform([visible_size, hidden_size], -r, r), name="weights2")
    biases2 = tf.Variable(tf.zeros([visible_size, 1]), name="biases2")
    
    # Add the variables to a collection so that they can be saved and restored.
    tf.add_to_collection('sparseAE', weights1)
    tf.add_to_collection('sparseAE', biases1)
    tf.add_to_collection('sparseAE', weights2)
    tf.add_to_collection('sparseAE', biases2)

    # Inference Operations
    a1 = images
    z2 = tf.matmul(weights1, tf.transpose(a1), name="z2") + biases1
    a2 = tf.sigmoid(z2)
    z3 = tf.matmul(weights2, a2, name="z3") + biases2
    a3 = tf.sigmoid(z3)
    # Loss computations

    m = images.get_shape().as_list()[0]
    rho_hat = tf.reduce_sum(a2, 1) / m
    cost_J = tf.reduce_sum(tf.square(tf.subtract(a3, tf.transpose(images)))) / (2 * m)
    cost_W = (tf.reduce_sum(tf.square(weights1)) + tf.reduce_sum(tf.square(weights2))) * decay / 2
    cost_KL = tf.reduce_sum(klterm(rho, rho_hat)) * beta
    cost = cost_J + cost_W + cost_KL
    # ------------------------------------------------------------------

  return cost


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
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def klterm(x, y):
  return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))