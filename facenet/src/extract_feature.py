import tensorflow as tf
import numpy as np
import facenet
import os
import time

from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_string('model_path', 'model/facenet_batch_1.pb',
                           """Directory containing the graph definition.""")
tf.app.flags.DEFINE_string('image_path', 'data/lfw-dlib-affine-sz-96/Aaron_Eckhart/Aaron_Eckhart_0001.png',
                           """The file containing the image.""")

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
    image_path = os.path.expanduser(FLAGS.image_path)

    with tf.Graph().as_default():
        # Creates graph from saved GraphDef
        #  NOTE: This does not work at the moment. Needs tensorflow to store variables in the graph_def.
        graphdef_filename = os.path.expanduser(FLAGS.model_path)
        f = gfile.FastGFile(graphdef_filename, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        images_placeholder, embeddings = tf.import_graph_def(graph_def, return_elements=['input', 'embeddings'],
                                                             name='')

        with tf.Session() as sess:
            paths = [image_path]
            input_tensor = sess.graph.get_tensor_by_name('input:0')
            output_tensor = sess.graph.get_tensor_by_name('embeddings:0')
            start_time = time.time()
            images = facenet.load_data(paths)
            feed_dict = {input_tensor: images}
            feature = sess.run([output_tensor], feed_dict=feed_dict)
            duration = time.time() - start_time
            print('Calculated embeddings for %s: time=%.3f seconds' % (image_path, duration))
            print('Feature: %s' % str(feature[0]))


if __name__ == '__main__':
    main()
