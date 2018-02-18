import tensorflow as tf
from tensorflow.python import debug as tf_debug
from loader import MNISTLoader
from estimator import Classifier
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class TempConfig(object):

    def __init__(self):
        self.batch_size = 128
        self.num_epochs = 3
        self.shuffle_buffer = 256
        self.data_dir = './dataset/'
        self.model_dir = './checkpoints/mnist/classify/'
        self.params = dict()
        self.params['hidden_units'] = [100, 100]
        self.params['n_classes'] = 10
        self.params['learning_rate'] = 0.01


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = TempConfig()
    loader = MNISTLoader(config)
    train_input_fn = loader.train_input_fn
    test_input_fn = loader.test_input_fn
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.train.MonitoredTrainingSession(config=sess_config)
    classifier = Classifier(config)
    estimator = classifier.get_estimator()

    estimator.train(input_fn=train_input_fn)
    eval_result = estimator.evaluate(input_fn=test_input_fn)
    print(eval_result)

if __name__=='__main__':
    main()
