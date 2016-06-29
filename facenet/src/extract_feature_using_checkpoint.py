import tensorflow as tf
import facenet
import os
import time

from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_string('model_dir', 'model/20160306-500000',
                           """Directory containing the graph definition and checkpoint files.""")
tf.app.flags.DEFINE_string('image_path', 'data/lfw-dlib-affine-sz-96/Aaron_Eckhart/Aaron_Eckhart_0001.png',
                           """The file containing the image.""")

tf.app.flags.DEFINE_integer('batch_size', 60,
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
        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')
        #
        # # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        #
        # # Build the inference graph
        embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, phase_train=phase_train_placeholder)

        # Create a saver for restoring variable averages
        ema = tf.train.ExponentialMovingAverage(1.0)
        saver = tf.train.Saver(ema.variables_to_restore())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(FLAGS.model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('Checkpoint not found')
            paths = [image_path] * FLAGS.batch_size
            start_time = time.time()
            images = facenet.load_data(paths)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            feature = sess.run([embeddings], feed_dict=feed_dict)
            duration = time.time() - start_time
            print('Calculated embeddings for %s: time=%.3f seconds' % (image_path, duration))
            print('Feature: %s' % str(feature[0]))


if __name__ == '__main__':
    main()
