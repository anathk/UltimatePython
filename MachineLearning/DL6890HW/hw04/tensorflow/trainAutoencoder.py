from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import tensorflow as tf
import math

import sparseAutoencoder as sparseAE
from displayNetwork import displayNetwork


def run_training(FLAGS, patches):
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    with tf.name_scope('input'):
      # Input data
      images_initializer = tf.placeholder(dtype=patches.dtype, shape=patches.shape)
      labels_initializer = tf.placeholder(dtype=patches.dtype, shape=patches.shape)
      input_images = tf.Variable(images_initializer, trainable=False, collections=[])
      input_labels = tf.Variable(labels_initializer, trainable=False, collections=[])

      image, label = tf.train.slice_input_producer([input_images, input_labels], num_epochs=FLAGS.num_epochs)
      image = tf.cast(image, tf.float32)
      label = tf.cast(label, tf.float32)
      images, labels = tf.train.batch([image, label], batch_size=FLAGS.batch_size)

    # Build a Graph that computes the loss for the sparse AutoEncoder.
    loss = sparseAE.loss(images, FLAGS.visibleSize, FLAGS.hiddenSize, FLAGS.decay, FLAGS.rho, FLAGS.beta)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = sparseAE.training(loss, FLAGS.learning_rate)

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
    sess.run(input_images.initializer, feed_dict={images_initializer: patches})
    sess.run(input_labels.initializer, feed_dict={labels_initializer: patches})

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

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
          saver.save(sess, FLAGS.log_dir + '/sparseAE', global_step=step)

        step += 1
    except tf.errors.OutOfRangeError:
      print('Saving')
      saver.save(sess, FLAGS.log_dir + '/sparseAE', global_step=step)
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    # Display learned filters.
    weights = [v for v in tf.trainable_variables() if 'sparseAE/weights1' in v.name][0]
    weights = weights.eval(session=sess)
    displayNetwork(weights, file_name = 'weights_tensor.jpg')

    # save the model before close the session
    saver_path = saver.save(sess, FLAGS.log_dir + "/" + FLAGS.saver_file)
    print("Model saved in file: %s" % saver_path)
    sess.close()

    return saver_path

