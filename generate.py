import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

from model import DCGAN


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/tensorflow/gan', 'Checkpoint directory')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')


FLAGS = tf.app.flags.FLAGS


def run_generator(saver, checkpoint_dir, model, random_z):
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, checkpoint)
        generated_images = model.visualize_generator()
        generated_images = sess.run(generated_images, feed_dict={model.random_z: random_z})
        return generated_images


def main(_):
    if not FLAGS.checkpoint_dir:
        raise ValueError('You must supply the checkpoint_path with --checkpoint_path')

    with tf.Graph().as_default():
        start_time = time.time()

        dcgan = DCGAN(mode='generate')
        dcgan.build()

        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        np.random.seed(0)
        random_z = np.random.uniform(-1, 1, [FLAGS.batch_size, 1, 1, 100])

        generated_images = run_generator(saver, FLAGS.checkpoint_dir, dcgan, random_z)

        print(generated_images.shape)

        img = generated_images[0]

        plt.imshow(img.astype(np.uint8))
        plt.axis('on')
        plt.title('Original image = RGB')
        plt.show()


if __name__ == '__main__':
    tf.app.run()