import tensorflow as tf
import os
import time

from model import DCGAN


tf.app.flags.DEFINE_string('train_dir', '/tmp/tensorflow/gan', 'Directory with trained model')
tf.app.flags.DEFINE_string('data_dir', '/Users/victor/Downloads/jpg', 'Directory with data')

tf.app.flags.DEFINE_integer('max_steps', 10000, 'Max steps')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')

tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 100, 'Epochs after which learning rate decays')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decay factor')


FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    if not FLAGS.data_dir:
        raise ValueError('data_dir not provided')

    if tf.gfile.Exists(FLAGS.train_dir):
        raise ValueError('This folder already exists.')
    tf.gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dcgan = DCGAN(FLAGS.data_dir, mode='train', batch_size=FLAGS.batch_size)
            dcgan.build()

            num_examples = len(os.listdir(FLAGS.data_dir))
            batches_per_epoch = num_examples / FLAGS.batch_size
            decay_steps = int(batches_per_epoch * FLAGS.num_epochs_per_decay)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                       global_step,
                                                       decay_steps=decay_steps,
                                                       decay_rate=FLAGS.learning_rate_decay_factor,
                                                       staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            opt_D = tf.train.AdamOptimizer(learning_rate, epsilon=1e-08)
            opt_G = tf.train.AdamOptimizer(learning_rate, epsilon=1e-08)

            optimize_D_op = opt_D.minimize(dcgan.discriminator_loss, global_step=global_step, var_list=dcgan.D_vars)
            optimize_G_op = opt_G.minimize(dcgan.generator_loss, global_step=global_step, var_list=dcgan.G_vars)

            variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)

            batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            batch_norm_updates_op = tf.group(*batch_norm_updates)

            train_op = tf.group(optimize_D_op, optimize_G_op, variables_averages_op, batch_norm_updates_op)

            with tf.control_dependencies([variables_averages_op, batch_norm_updates_op]):
                optimize_D_op
                optimize_G_op

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

            init = tf.global_variables_initializer()
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            summary_op = tf.summary.merge_all()

            for step in range(FLAGS.max_steps + 1):
                start_time = time.time()

                _, step_val, loss_D, loss_G = sess.run([train_op, global_step, dcgan.discriminator_loss, dcgan.generator_loss])

                epochs = step * FLAGS.batch_size / num_examples

                duration = time.time() - start_time

                if step % 10 == 0:
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    print("Epochs: %.2f step: %d  loss_D: %f loss_G: %f (%.1f examples/sec; %.3f sec/batch)"
                          % (epochs, step, loss_D, loss_G, examples_per_sec, duration))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % 200 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    tf.app.run()