# ==============================================================================
# DL6890 Self-taught Learning Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  self-taught learning. You will need to complete code in feedForwardAutoencoder.py
#  For sparseAutoencoder.py and softmax.py, you are encouraged to reuse code
#  from the previous assignments.
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

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import trainAutoencoder
import trainSoftmax
import softmax
from feedForwardAutoencoder import feedForwardAutoencoder


# Set parameters for Sparse Autoencoder
parser = argparse.ArgumentParser('Sparse AutoEncoder Exercise.')
parser.add_argument('--learning_rate', 
                    type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=400, 
                    help='Number of epochs to run trainer.')
parser.add_argument('--batch_size', 
                    type=int, default=100, 
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--input_data_dir', 
                    type=str, 
                    default='../mnist/data', 
                    help='Directory to put the training data.')
parser.add_argument('--log_dir', 
                    type=str, 
                    default='logs', 
                    help='Directory to put logging.')
parser.add_argument('--saver_file', 
                    type=str, 
                    default='sparseAE.ckpt', 
                    help='File to save checkpoint with sAE model.')
parser.add_argument('--visibleSize',
                    type=int,
                    default=str(28 * 28),
                    help='Used for gradient checking.')
parser.add_argument('--hiddenSize', 
                    type=int,
                    default='200',
                    help='.')
parser.add_argument('--rho', 
                    type=float,
                    default='0.1',
                    help='Sparsity parameter.')
parser.add_argument('--decay', 
                    type=float,
                    default='3e-3',
                    help='.')
parser.add_argument('--beta', 
                    type = float,
                    default='3',
                    help='')
 
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()

# ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# load mnist data set
data_sets = input_data.read_data_sets(FLAGS.input_data_dir)

# only use the train set of MNIST for this exercise
raw_images = data_sets.train.images
raw_labels = data_sets.train.labels

# use 5 - 9 digits to train sparse autoencoder
unlabeled_images = [image for (image, label) in zip(raw_images, raw_labels) if label >= 5]
unlabeled_images = numpy.asarray(unlabeled_images)

# used 0 - 4 digits to train and evaluate a softmax classifier
labeled_images = [image for (image, label) in zip(raw_images, raw_labels) if label <= 4]
labeled_images = numpy.asarray(labeled_images)
labeled_labels = [label for (image, label) in zip(raw_images, raw_labels) if label <= 4]
labeled_labels = numpy.asarray(labeled_labels)

num_train = labeled_images.shape[0] // 2

# first half of the 0 - 4 digits sets is training set
train_images = labeled_images[0:num_train, :]
train_labels = labeled_labels[0:num_train]

# second half of the 0 - 4 digits sets is testing set
test_images  = labeled_images[num_train:, :]
test_labels  = labeled_labels[num_train:]

# Output Some Statistics
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
#  This trains the sparse autoencoder on the unlabeled training images,
#  and save the trained parameters to disk.
saver_path = trainAutoencoder.run_training(FLAGS, unlabeled_images)

# ======================================================================
#  STEP 3: Extract Features from the Supervised Dataset

# ----------------- YOUR CODE HERE ----------------------
#  Use feedForwardAutoencoder to extract featuers from train and test images
#  You need to complete the code in feedForwardAutoencoder.py.

train_features = feedForwardAutoencoder(saver_path, train_images)
test_features = feedForwardAutoencoder(saver_path, test_images)

# ======================================================================
#  STEP 4: Train the softmax classifier
#

# Set parameters for softmax classifier
FLAGS.learning_rate = 0.1
FLAGS.visibleSize = train_features.shape[1]
FLAGS.max_steps = 80000
FLAGS.batch_size = 100

# Utilize the built-in DataSet to create datasets for training and testing.
# Set reshape = False and dtype = tf.uint8 to avoid changing of the features
# by DataSet.
train_set = DataSet(train_features, train_labels, reshape=False, dtype=tf.uint8)
test_set = DataSet(test_features, test_labels, reshape=False, dtype=tf.uint8)

#  Use softmax.py from the previous assignment to train the classifier.
saver_path = trainSoftmax.run_training(FLAGS, train_set, test_set)

# ======================================================================
#  STEP 4: Testing 
#
# ----------------- YOUR CODE HERE ----------------------
# Compute accuracy of predictions on the test set (testFeatures) using the
# softmaxModel saved in 'saver_path'. Use softmaxPredict() from softmax.py.
correct = softmax.softmaxPredict(saver_path, test_features, test_labels)
total = test_labels.shape[0]
acc = correct/total

# Classification Score
print()
print(20*'*', 'the accuracy of the trained model', 20*'*')
print(acc)
print()

#
# Accuracy is the proportion of correctly classified images
# The results for our implementation was:
#
# Accuracy: 98.2%
#

