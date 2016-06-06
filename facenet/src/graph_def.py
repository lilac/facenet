import tensorflow as tf
import numpy as np
import facenet
import os
import time

from tensorflow.python.client import graph_util
# from tensorflow.python.platform import gfile
tf.app.flags.DEFINE_string('model_dir', '~/models/facenet/20160501-133835',
                           """Directory containing the graph definition and checkpoint files.""")
tf.app.flags.DEFINE_string("output_dir", ".",
                           """Output 'GraphDef' dir""")
tf.app.flags.DEFINE_string("output_name", "graph_def.pb",
                           """Output 'GraphDef' file name.""")

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_boolean('random_crop', False,
                            """Performs random cropping of training images. If false, the center image_size pixels from the training images are used.
                            If the size of the images in the data directory is equal to image_size no cropping is performed""")
tf.app.flags.DEFINE_boolean('random_flip', False,
                            """Performs random horizontal flipping of training images.""")
tf.app.flags.DEFINE_float('keep_probability', 1.0,
                          """Keep probability of dropout for the fully connected layer(s).""")
tf.app.flags.DEFINE_integer('seed', 666,
                            """Random seed.""")

FLAGS = tf.app.flags.FLAGS


def main():

    with tf.Graph().as_default():

        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3),
                                            name='input')

        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # Build the inference graph
        embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, phase_train=tf.constant(False))

        # Create a saver for restoring variable averages
        ema = tf.train.ExponentialMovingAverage(1.0)
        saver = tf.train.Saver(ema.variables_to_restore())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(FLAGS.model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('Checkpoint not found')

            graphdef_filename = FLAGS.output_name
            print('Saving graph definition')
            output_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), ['embeddings'])

            tf.train.write_graph(output_graph_def, FLAGS.output_dir, graphdef_filename, False)


if __name__ == '__main__':
    main()