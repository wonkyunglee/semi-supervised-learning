#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function



import tensorflow as tf
from tensorflow.python import debug as tf_debug
from loader import MNISTLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class TempConfig(object):

    def __init__(self):
        self.batch_size = 60000
        self.num_epochs = 3
        self.shuffle_buffer = 60000
        self.data_dir = './dataset/'


class Tests(tf.test.TestCase):

    def test_loader(self):

        config = TempConfig()
        loader = MNISTLoader(config)
        input_fn = loader.train_input_fn()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.train.MonitoredTrainingSession(config=sess_config)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        i = 0
        while True:
            try:
                images, labels = sess.run(input_fn)
                print(i, images.shape, labels.shape)
                i += 1
                self.assertEqual(images.shape, (config.batch_size, 28*28))
                self.assertEqual(labels.shape, (config.batch_size, ))
            except tf.errors.OutOfRangeError:
                self.assertEqual(i, config.num_epochs)
                break


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.test.main()
