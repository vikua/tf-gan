import os
import tensorflow as tf


def read_images(input_queue):
    file_contents = tf.read_file(input_queue.dequeue())
    example = tf.image.decode_jpeg(file_contents, channels=3)

    # transform RGBs 0..255 to 0..1
    example = tf.image.convert_image_dtype(example, dtype=tf.float32)

    # resizing and distortion
    example = tf.image.resize_images(example, [64, 64])
    example.set_shape((64, 64, 3))
    distorted_example = tf.image.random_flip_left_right(example)

    # transform 0..1 to -1..1
    image = tf.subtract(distorted_example, 0.5)
    image = tf.multiply(image, 2.0)

    return image


def input_pipeline(data_dir, batch_size=32, read_threads=4):
    with tf.name_scope('batch_processing'):
        data_files = tf.gfile.Glob(os.path.join(data_dir, '*'))
        filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        example_list = [[read_images(filename_queue)] for _ in range(read_threads)]

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        image_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)
        return image_batch