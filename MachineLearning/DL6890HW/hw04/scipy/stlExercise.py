# ==============================================================================
# DL6890 Self-taught Learning Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the self-taught learning
#  assignment.
#  You will need to write code in:
#    feedForwardAutoencoder.py
#  You will need code from the previous assignments in:
#    sparseAutoencoder.py
#    softmax.py
#
# ======================================================================
#  STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import math

import numpy as np
from numpy.linalg import norm
from numpy.random import randint, uniform
from scipy.optimize import fmin_l_bfgs_b

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

from sparseAutoencoder import sparseAutoencoderCost
from computeNumericalGradient import computeNumericalGradient
from displayNetwork import displayNetwork

import trainAutoencoder
import trainSoftmax
from feedForwardAutoencoder import feedForwardAutoencoder
from softmax import softmaxPredict


# Set parameters for Sparse Autoencoder
parser = argparse.ArgumentParser('Sparse AutoEncoder Exercise.')
parser.add_argument('-t', '--input_type',
                    type=str,
                    choices = ['natural', 'digits'],
                    default='digits',
                    help = 'Type of images used for training.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../mnist/data',
                    help='Directory to put the input MNIST data.')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Used for gradient checking.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=400,
                    help='Number of epochs to run trainer.')
parser.add_argument('--log_dir',
                    type=str,
                    default='logs',
                    help='Directory to put logging.')
parser.add_argument('--params_file',
                    type=str,
                    default='sparseAE.pkl',
                    help='File name for saving sAE parameters.')
parser.add_argument('--visibleSize',
                    type=int,
                    default=str(28 * 28),
                    help='Used for gradient checking.')
parser.add_argument('--hiddenSize', 
                    type=int,
                    default='200',
                    help='neuron number of hidden layer.')
parser.add_argument('--rho', 
                    type=float,
                    default='0.1',
                    help='Sparsity parameter.')
parser.add_argument('--decay', 
                    type=float,
                    default='3e-3',
                    help='Penalty weight for regularization.')
parser.add_argument('--beta', 
                    type = float,
                    default='3',
                    help='Panalty weight for sparsity.')
 
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()

 
# ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# Load MNIST data set.
data_sets = input_data.read_data_sets(FLAGS.input_data_dir)

# Use only the train set of MNIST for this exercise.
raw_images = data_sets.train.images
raw_labels = data_sets.train.labels

# Use unlabeled digits 5 - 9 to train autoencoder.
unlabeled_images = [image for (image, label) in zip(raw_images, raw_labels) if label >= 5]
unlabeled_images = np.asarray(unlabeled_images)

# Use digits 0 - 4 digits to train and evaluate a softmax classifier.
labeled_images = [image for (image, label) in zip(raw_images, raw_labels) if label <= 4]
labeled_images = np.asarray(labeled_images)
labeled_labels = [label for (image, label) in zip(raw_images, raw_labels) if label <= 4]
labeled_labels = np.asarray(labeled_labels)

num_train = labeled_images.shape[0] // 2

# First half of the 0 - 4 digits dataset is used for training.
train_images = labeled_images[0:num_train, :]
train_labels = labeled_labels[0:num_train]

# Second half of the 0 - 4 digits dataset is used for testing.
test_images  = labeled_images[num_train:, :]
test_labels  = labeled_labels[num_train:]

# Print some statistics.
print()
print(20 * '*', 'examples in unlabeled set', 20 * '*') 
print(unlabeled_images.shape[0])
print()
print(20 * '*', 'examples in supervised training set', 20 * '*')
print(train_images.shape[0])
print()
print(20 * '*', 'examples in supervised testing set', 20 * '*')
print(test_images.shape[0])
print()

# ======================================================================
#  STEP 2: Train the sparse autoencoder
#
#  This trains the sparse autoencoder on the unlabeled training images
#  and saves the trained parameters to disk for later use.
trainAutoencoder.run_training(FLAGS, unlabeled_images)

# ======================================================================
#  STEP 3: Extract Features from the Supervised Dataset
#  
#  You need to complete the code in feedForwardAutoencoder.py and use it
#  here to extract features from train and test images.


# ----------------- YOUR CODE HERE ----------------------

trainFeatures =

testFeatures = 

# ======================================================================
#  STEP 4: Train the softmax classifier

#  Set parameters for Softmax classifier
FLAGS.visibleSize = train_features.shape[0]
FLAGS.decay = 1e-4

#  The trainSoftmax.py uses softmax.py from the previous assignment to train
#  a multi-class classifier on trainFeatures and trainLabels.

theta = trainSoftmax.run_training(FLAGS, train_features, train_labels)

# ======================================================================
#  STEP 4: Testing 
#
# ----------------- YOUR CODE HERE ----------------------
# Compute accuracy of predictions on the test set (testFeatures) using
# softmax. You will have to implement softmaxPredict in softmax.py.

acc = 

# Print classification accuracy.
print(20 * '*', 'the accuracy of the trained model', 20 * '*')
print('%0.3f%%.' % (acc * 100))
print()

#
# Accuracy is the proportion of correctly classified images
# The results for our implementation was:
#
# Accuracy: 98.2%
#

