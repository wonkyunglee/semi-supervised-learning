#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from loader import MNISTLoader
#from estimator import Classifier
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class TempConfig(object):

    def __init__(self):
        self.batch_size = 60000
        self.num_epochs = 3
        self.shuffle_buffer = 60000
        self.data_dir = './dataset/'
        self.params = dict()
        self.params['hidden_units'] = [100, 100]
        self.params['n_classes'] = 10
        self.learning_rate = 0.01


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
                images = images['x']
                print(i, images.shape, labels.shape)
                i += 1
                self.assertEqual(images.shape, (config.batch_size, 28*28))
                self.assertEqual(labels.shape, (config.batch_size, ))
            except tf.errors.OutOfRangeError:
                self.assertEqual(i, config.num_epochs)
                break


#   def test_estimator(self):

#        config = TempConfig()
#        loader = MNISTLoader(config)
#        train_input_fn = loader.train_input_fn()
#        test_input_fn = loader.test_input_fn()
#        sess_config = tf.ConfigProto()
#        sess_config.gpu_options.allow_growth = True
#        sess = tf.train.MonitoredTrainingSession(config=sess_config)
#        classifier = Classifier(config).get_estimator()
#
#        classifier.train(input_fn=train_input_fn)
#        eval_result = classifier.evaluate(input_fn=test_input_fn)
#        print(eval_result)
#        #self.assertEqual(3, config.num_epochs)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.test.main()
