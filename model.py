import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
import image_utils

batch_size = 32
random_z_size = 100


class DCGAN(object):
    def __init__(self, data_dir=None, mode='train', batch_size=16, random_z_size=100, num_threads=4):
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size
        self.random_z_size = random_z_size
        self.num_threads = num_threads

    def leakyrelu(self, x, leaky_weight=0.2, name=None):
        with ops.name_scope(name, "LRelu", [x]) as name:
            return tf.maximum(x, leaky_weight * x)

    def create_random_z(self):
        with tf.variable_scope('random_z'):
            if self.mode == 'generate':
                self.random_z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, random_z_size], name='random_z')
            else:
                self.random_z = tf.random_uniform([batch_size, 1, 1, random_z_size], minval=-1, maxval=1, dtype=tf.float32)
        return self.random_z

    def generator(self, random_z, is_training=True):
        with tf.variable_scope('Generator'):
            batch_norm_params = {'decay': 0.999, 'epsilon': 0.001, 'is_training': is_training, 'scope': 'batch_norm'}
            with tf.contrib.framework.arg_scope([layers.conv2d_transpose],
                                                kernel_size=[4, 4],
                                                stride=[2, 2],
                                                normalizer_fn=layers.batch_norm,
                                                normalizer_params=batch_norm_params,
                                                weights_regularizer=layers.l2_regularizer(0.00004, scope='l2_decay')):
                # input: 1 x 1 x 100
                self.conv1 = layers.conv2d_transpose(inputs=random_z,
                                                     num_outputs=64 * 8,
                                                     padding='VALID',
                                                     scope='conv1')

                # conv1: 4 x 4 x (64 * 8)
                self.conv2 = layers.conv2d_transpose(inputs=self.conv1,
                                                     num_outputs=64 * 4,
                                                     scope='conv2')

                # conv2: 8 x 8 x (64 * 4)
                self.conv3 = layers.conv2d_transpose(inputs=self.conv2,
                                                     num_outputs=64 * 2,
                                                     scope='conv3')

                # conv3: 16 x 16 x (64 * 2)
                self.conv4 = layers.conv2d_transpose(inputs=self.conv3,
                                                     num_outputs=64,
                                                     scope='conv4')

                # conv4: 32 x 32 x 64
                self.conv5 = layers.conv2d_transpose(inputs=self.conv4,
                                                     num_outputs=3,
                                                     normalizer_fn=None,
                                                     biases_initializer=None,
                                                     activation_fn=tf.tanh,
                                                     scope='conv5')

                # output 64 x 64 x 3
                return self.conv5

    def discriminator(self, images, reuse=False):
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.999, 'epsilon': 0.001, 'scope': 'batch_norm'}
            with tf.contrib.framework.arg_scope([layers.conv2d],
                                                kernel_size=[4, 4],
                                                stride=[2, 2],
                                                activation_fn=self.leakyrelu,
                                                normalizer_fn=layers.batch_norm,
                                                normalizer_params=batch_norm_params,
                                                weights_regularizer=layers.l2_regularizer(0.00004, scope='l2_decay')):
                # input: 64 x 64 x 3
                self.conv1 = layers.conv2d(inputs=images,
                                           num_outputs=64,
                                           normalizer_fn=None,
                                           biases_initializer=None,
                                           scope='conv1')

                # conv1: 32 x 32 x 64
                self.conv2 = layers.conv2d(inputs=self.conv1,
                                           num_outputs=64 * 2,
                                           scope='conv2')

                # conv2: 16 x 16 x (64 * 2)
                self.conv3 = layers.conv2d(inputs=self.conv2,
                                           num_outputs=64 * 4,
                                           scope='conv3')

                # conv3: 8 x 8 x (64 * 4)
                self.conv4 = layers.conv2d(inputs=self.conv3,
                                           num_outputs=64 * 8,
                                           scope='conv4')

                # conv4: 4 x 4 x (64 * 8)
                self.conv5 = layers.conv2d(inputs=self.conv4,
                                           num_outputs=1,
                                           stride=[1, 1],
                                           padding='VALID',
                                           normalizer_fn=None,
                                           normalizer_params=None,
                                           activation_fn=None,
                                           scope='conv5')
                return self.conv5

    def get_real_images(self):
        with tf.variable_scope('real_images'):
            real_images = image_utils.input_pipeline(data_dir=self.data_dir,
                                                     batch_size=self.batch_size,
                                                     read_threads=self.num_threads)
            return real_images

    def gan_loss(self, logits, real=True, smoothing=0.9, name=None):
        if real:
            labels = tf.fill(logits.get_shape(), smoothing)
            # labels = tf.ones_like(logits)
        else:
            labels = tf.zeros_like(logits)

        with ops.name_scope(name, 'GAN_loss', [logits, labels]) as name:
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits))
            return loss

    def build(self):
        random_z = self.create_random_z()

        if self.mode == 'generate':
            self.generated_images = self.generator(random_z, is_training=False)

        if self.mode == 'train':
            self.generated_images = self.generator(random_z)
            self.real_images = self.get_real_images()

            self.real_logits = self.discriminator(self.real_images)
            self.fake_logits = self.discriminator(self.generated_images, reuse=True)

            self.real_loss = self.gan_loss(self.real_logits, real=True)
            self.fake_loss = self.gan_loss(self.fake_logits, real=False)

            self.discriminator_loss = self.real_loss + self.fake_loss
            self.generator_loss = self.gan_loss(self.fake_logits, real=True)

            t_vars = tf.trainable_variables()
            self.D_vars = [var for var in t_vars if 'Discriminator' in var.name]
            self.G_vars = [var for var in t_vars if 'Generator' in var.name]

            for var in self.G_vars:
                print(var)
            for var in self.D_vars:
                print(var)

            tf.summary.scalar('losses/loss_discriminator', self.discriminator_loss)
            tf.summary.scalar('losses/loss_generator', self.generator_loss)
            tf.summary.scalar('losses/loss_real', self.real_loss)
            tf.summary.scalar('losses/loss_fake', self.fake_loss)

            tf.summary.image('random_images', self.generated_images, max_outputs=10)
            tf.summary.image('real_images', self.real_images, max_outputs=10)

    def visualize_generator(self):
        generated_images = tf.add(self.generated_images, 1.0)
        generated_images = tf.multiply(generated_images, 0.5 * 255.0)
        generated_images = tf.to_int32(generated_images)
        return generated_images
