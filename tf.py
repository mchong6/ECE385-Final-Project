
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pylab
from scipy import misc


import tensorflow as tf

FLAGS = None


def main():
  # Import data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # Create the model
  #x = tf.placeholder(tf.float32, [None, 784])
  #W = tf.Variable(tf.zeros([784, 30]))
  #W2 = tf.Variable(tf.zeros([30, 10]))
  #b = tf.Variable(tf.zeros([30]))
  #b2 = tf.Variable(tf.zeros([10]))
  #y1 = tf.sigmoid(tf.matmul(x, W) + b)
  #y = tf.sigmoid(tf.matmul(y1, W2) + b2)

  train = 0
  x = tf.placeholder(tf.float32, [None, 784])
  if train:
      W = tf.Variable(tf.zeros([784, 10]))
      #b = tf.Variable(tf.zeros([10]))
  else:
      W = tf.Variable(np.around(np.loadtxt("test.csv", dtype='float32',delimiter=',')))
      #b = tf.Variable(np.around(np.loadtxt("test2.csv", dtype='float32',delimiter=',')))

  y = (tf.matmul(x, W))
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  regularize = 0.001 * tf.nn.l2_loss(W)
  #MSE_loss = reduce_sum(tf.squared_difference(y_, y))
  #train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy+regularize)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(60000):
    if (train):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #convert to black-white 
        batch_xs = 1. * (batch_xs > 0.5)
        test_image = batch_xs[0,:].reshape([28,28])
        plt.imshow(test_image)
        pylab.show()
        sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  bw_test = 1. * (mnist.test.images > 0.5)
  test_image = misc.imread('temp.png').reshape([1,784])
  test_image2 = misc.imread('temp.png')
  test_image = 1. *(test_image > 0.5)
  test_image2 = 1. *(test_image2 > 0.5)
  #test_image = bw_test[0,:].reshape([28,28])
  #test_image = bw_test[0,:].reshape([1,784])
  print(test_image)
  #print(test_image)
  plt.imshow(test_image2)
  pylab.show()
  #sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
  #print(sess.run(accuracy, feed_dict={x: bw_test,
  #                                    y_: mnist.test.labels}))
  print(sess.run(y, feed_dict={x: test_image,
                                      y_: mnist.test.labels}))
  if train:
      np.savetxt("test.csv", (np.around(W.eval() * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
      #np.savetxt("test2.csv", (np.around(b.eval() * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
main()
