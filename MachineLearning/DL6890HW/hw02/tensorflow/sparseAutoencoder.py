"""Builds the sAE network and operations to compute the cost.

Implements the loss/training pattern for feature learning.

1. los() - Builds the model as far as is required for running the network
forward to make predictions. Adds to the inference model the layers required
to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the "sparseAutoencoderExercise.py" file and not meant to be run.
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
  # But initialize biases with zeros.
  r  = math.sqrt(6) / math.sqrt(hidden_size + visible_size + 1)

  with tf.name_scope('sparseAE'):
    ## ---------- YOUR CODE HERE --------------------------------------
    # Define Inference Variables, call first layer weights 'weights1'.
    # Weights
    weights1 = tf.Variable(tf.random_uniform([visible_size, hidden_size], minval=-r, maxval=r), name='weights1')
    weights2 = tf.Variable(tf.random_uniform([hidden_size, visible_size], minval=-r, maxval=r))
    bias1 = tf.Variable(tf.zeros([hidden_size]))
    bias2 = tf.Variable(tf.zeros([visible_size]))

    # Define Inference Operations
    a2 = tf.sigmoid(tf.add(tf.matmul(images, weights1), bias1))
    a3 = tf.sigmoid(tf.add(tf.matmul(a2, weights2), bias2))


    rho_hat = tf.reduce_mean(a2, axis=0)
    # Loss computations
    # squared_error = np.sum((h - data) ** 2) / (2 * m)
    squared_error = tf.reduce_sum(tf.reduce_mean(tf.square(a3 - images), axis=0))
    # weight_decay = (decay / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    weight_decay = decay * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
    # sparsity_term = beta * np.sum(sparse_rho * np.log(sparse_rho / rho_hat) + (1 - sparse_rho) * np.log((1 - sparse_rho) / (1 - rho_hat)))
    sparsity_term = beta * (tf.reduce_sum(rho * tf.log(rho / rho_hat) + (1 - rho) * tf.log((1 - rho) / (1 - rho_hat))))

    cost = squared_error + weight_decay + sparsity_term
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
