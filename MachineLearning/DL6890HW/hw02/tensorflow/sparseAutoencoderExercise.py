"""Trains a sparse autoencoder on digits or natural images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf

import sparseAutoencoder
from displayNetwork import displayNetwork

# Basic model parameters as external flags.
FLAGS = None


def run_training():
  """Train sAE for a number of epochs."""

  # Get the sets of images and for training
  numPatches = 10000
  if FLAGS.input_type == 'natural':
    from sampleNaturalImages import sampleNaturalImages
    patches = sampleNaturalImages('IMAGES.mat', numPatches)
    epochs = 4 * FLAGS.num_epochs
  else:
    from sampleDigitImages import sampleDigitImages
    patches = sampleDigitImages(FLAGS.train_dir, 2 * numPatches)
    epochs = FLAGS.num_epochs

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    with tf.name_scope('input'):
      # Input data
      images_initializer = tf.placeholder(
          dtype=patches.dtype,
          shape=patches.T.shape)
      labels_initializer = tf.placeholder(
          dtype=patches.dtype,
          shape=patches.T.shape)
      input_images = tf.Variable(
          images_initializer, trainable=False, collections=[])
      input_labels = tf.Variable(
          labels_initializer, trainable=False, collections=[])

      image, label = tf.train.slice_input_producer(
          [input_images, input_labels], num_epochs=epochs)
      image = tf.cast(image, tf.float32)
      label = tf.cast(label, tf.float32)
      images, labels = tf.train.batch(
          [image, label], batch_size=FLAGS.batch_size)

    # Build a Graph that computes the loss for the sparse AutoEncoder.
    loss = sparseAutoencoder.loss(images, FLAGS.visibleSize, FLAGS.hiddenSize, FLAGS.decay, FLAGS.rho, FLAGS.beta)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = sparseAutoencoder.training(loss, FLAGS.learning_rate)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create the op for initializing variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(init_op)
    sess.run(input_images.initializer,
             feed_dict={images_initializer: patches.T})
    sess.run(input_labels.initializer,
             feed_dict={labels_initializer: patches.T})

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # And then after everything is built, start the training loop.
    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()

        # Run one step of the model.
        _, loss_value = sess.run([train_op, loss])

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
          # Print status to stdout.
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
          # Update the events file.
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)
          step += 1

        # Save a checkpoint periodically.
        if (step + 1) % 5000 == 0:
          print('Saving')
          saver.save(sess, FLAGS.train_dir, global_step=step)

        step += 1
    except tf.errors.OutOfRangeError:
      print('Saving')
      saver.save(sess, FLAGS.train_dir, global_step=step)
      print('Done training for %d epochs, %d steps.' % (epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    # Display learned filters.
    weights = [v for v in tf.trainable_variables() if 'sparseAE/weights1' in v.name][0]
    weights = weights.eval(session=sess)
    # weights = sess.run(weights)
    displayNetwork(weights, file_name = 'weights-' + FLAGS.input_type + '.jpg')

    sess.close()


def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Sparse AutoEncoder Exercise.')
  parser.add_argument('--input_type',
                      type=str,
                      choices = ['natural', 'digits'],
                      default='natural',
                      help = 'Type of images used for training.')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=100,
      help='Number of epochs to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='../../mnist/data',
      help='Directory to put the training data.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  parser.add_argument('--visibleSize', type=int)
  parser.add_argument('--hiddenSize', type=int)
  parser.add_argument('--rho', type=float)
  parser.add_argument('--decay', type=float)
  parser.add_argument('--beta', type = float)
  if FLAGS.input_type == 'natural':
    parser.parse_args(args=['--visibleSize', str(8*8)], namespace=FLAGS)
    parser.parse_args(args=['--hiddenSize', '25'], namespace=FLAGS)
    parser.parse_args(args=['--rho', '0.01'], namespace=FLAGS)
    parser.parse_args(args=['--decay', '0.0001'], namespace=FLAGS)
    parser.parse_args(args=['--beta', '3'], namespace=FLAGS)
  else:
    parser.parse_args(args=['--visibleSize', str(28*28)], namespace=FLAGS)
    parser.parse_args(args=['--hiddenSize', '196'], namespace=FLAGS)
    parser.parse_args(args=['--rho', '0.1'], namespace=FLAGS)
    parser.parse_args(args=['--decay', '3e-3'], namespace=FLAGS)
    parser.parse_args(args=['--beta', '3'], namespace=FLAGS)

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
