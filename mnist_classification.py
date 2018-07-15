from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=False)

  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 3)

  with tf.name_scope('input_downsample'):
    image_ds_input = tf.image.resize_images(image_shaped_input, (14,14))
    tf.summary.image('downsample_input', image_ds_input, 3)

  with tf.name_scope('input_blur'):
    blur_filter = (1/9) * np.ones((3,3))
    blur_filter_tf = tf.constant(blur_filter, dtype=tf.float32, shape=[3,3,1,1], name='blur_filter')
    image_blur_input = tf.nn.conv2d(image_ds_input, blur_filter_tf, strides=[1,1,1,1], padding='SAME', name='blurring')
    tf.summary.image('blur_input', image_blur_input, 3)

  def model(input, is_training, L, N, keep_prob):
      y = tf.layers.conv2d(input, N, 3, (2,2), padding='same', name='conv2d_%d_0'% L, activation=tf.nn.relu)
      y = tf.layers.batch_normalization(y, training=is_training, name='BN_%d_%d' % (L,0))
      for l in range(1,L,1):
          y = tf.layers.conv2d(y, N, 3, (2,2), padding='same', name='conv2d_%d_%d' % (L,l), activation=tf.nn.relu)
          y = tf.layers.batch_normalization(y, training=is_training, name='BN_%d_%d' % (L,l))

##      tf.summary.image('conv_out_%d_0' % L, tf.expand_dims(y[:,:,:,0], 3))
##      tf.summary.image('conv_out_%d_1' % L, tf.expand_dims(y[:,:,:,1], 3))
##      tf.summary.image('conv_out_%d_2' % L, tf.expand_dims(y[:,:,:,2], 3))
##      tf.summary.image('conv_out_%d_3' % L, tf.expand_dims(y[:,:,:,3], 3))

      _, h, w, c = y.shape
      y = tf.reshape(y, [-1, h*w*c], name='reshape_%d' % L)

      y = tf.layers.dense(y, h*w*c, activation=tf.nn.relu , name='fc_%d_0' % L)

      y = tf.layers.dropout(y, rate=keep_prob, training=is_training, name='dropout_%d' % L)

      y = tf.layers.dense(y, 2   , activation=tf.identity, name='fc_%d_1' % L)

      return y
  
  is_training = tf.placeholder(tf.bool, name='is_trianing')
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
  N = FLAGS.N
  y1 = model(image_blur_input, is_training, 1, N, keep_prob)
  y2 = model(image_blur_input, is_training, 2, N, keep_prob)
  y3 = model(image_blur_input, is_training, 3, N, keep_prob)

  with tf.name_scope('cross_entropy1'):
    with tf.name_scope('total1'):
      cross_entropy1 = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y1)
  with tf.name_scope('cross_entropy2'):
    with tf.name_scope('total2'):
      cross_entropy2 = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y2)
  with tf.name_scope('cross_entropy3'):
    with tf.name_scope('total3'):
      cross_entropy3 = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y3)
  tf.summary.scalar('cross_entropy1', cross_entropy1)
  tf.summary.scalar('cross_entropy2', cross_entropy2)
  tf.summary.scalar('cross_entropy3', cross_entropy3)

  tf_learning_rate = tf.placeholder(tf.float32, name='lr')
  tf.summary.scalar('learning_rate', tf_learning_rate)
  with tf.name_scope('train1'):
    train_step1 = tf.train.AdamOptimizer(tf_learning_rate).minimize(cross_entropy1)
  with tf.name_scope('train2'):
    train_step2 = tf.train.AdamOptimizer(tf_learning_rate).minimize(cross_entropy2)
  with tf.name_scope('train3'):
    train_step3 = tf.train.AdamOptimizer(tf_learning_rate).minimize(cross_entropy3)

  with tf.name_scope('accuracy1'):
    with tf.name_scope('correct_prediction1'):
      correct_prediction1 = tf.equal(tf.argmax(y1, 1), y_)
    with tf.name_scope('accuracy1'):
      accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

  with tf.name_scope('accuracy2'):
    with tf.name_scope('correct_prediction2'):
      correct_prediction2 = tf.equal(tf.argmax(y2, 1), y_)
    with tf.name_scope('accuracy2'):
      accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

  with tf.name_scope('accuracy3'):
    with tf.name_scope('correct_prediction3'):
      correct_prediction3 = tf.equal(tf.argmax(y3, 1), y_)
    with tf.name_scope('accuracy3'):
      accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

  tf.summary.scalar('accuracy1', accuracy1)
  tf.summary.scalar('accuracy2', accuracy2)
  tf.summary.scalar('accuracy3', accuracy3)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train_%d' % FLAGS.N, sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test_%d' % FLAGS.N)
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train, lr):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100, fake_data=False)
      is_tr = True
      k = FLAGS.dropout

    else:
      xs, ys = mnist.test.images, mnist.test.labels
      is_tr = False
      k = 1.0

    ys2 = np.zeros((ys.shape), dtype=np.int64)
    ys2[ys >= 3] = 1
    ys2[ys <  3] = 0
    
    return {x: xs, y_: ys2, is_training: is_tr, keep_prob:k, tf_learning_rate: lr}

  for i in range(FLAGS.max_steps):
    lr = FLAGS.learning_rate
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc1, acc2, acc3 = sess.run([merged, accuracy1, accuracy2, accuracy3], feed_dict=feed_dict(False, lr))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %6s, %6s, %6s' % (i, acc1, acc2, acc3))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary1, _ = sess.run([merged, train_step1], feed_dict=feed_dict(True, lr), options=run_options, run_metadata=run_metadata)
        summary2, _ = sess.run([merged, train_step2], feed_dict=feed_dict(True, lr), options=run_options, run_metadata=run_metadata)
        summary3, _ = sess.run([merged, train_step3], feed_dict=feed_dict(True, lr), options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary1, i)
        train_writer.add_summary(summary2, i)
        train_writer.add_summary(summary3, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary1, _ = sess.run([merged, train_step1], feed_dict=feed_dict(True, lr))
        summary2, _ = sess.run([merged, train_step2], feed_dict=feed_dict(True, lr))
        summary3, _ = sess.run([merged, train_step3], feed_dict=feed_dict(True, lr))
        train_writer.add_summary(summary1, i)
        train_writer.add_summary(summary2, i)
        train_writer.add_summary(summary3, i)
  train_writer.close()
  test_writer.close()

def main(_):
  #if tf.gfile.Exists(FLAGS.log_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.log_dir)
  #tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps to run trainer.')
  parser.add_argument('--N', type=int, default=8, help='Number of CNN layers.')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='input_data', help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='logs', help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
