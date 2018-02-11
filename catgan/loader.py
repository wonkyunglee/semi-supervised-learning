import tensorflow as tf
import urllib
import gzip
import shutil
import os

class MNISTLoader(object):
    """download, decode_image, decod_label methods are copied from
    https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py
    """
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_parallel_calls = config.num_parallel_calls
        self.shuffle_buffer = config.shuffle_buffer
        self.data_dir = config.data_dir


    def get_dataset(self, images_filename, labels_filename):
        images_filepath = self.download(self.data_dir, images_filename)
        labels_filepath = self.download(self.data_dir, labels_filename)

        def decode_image(image):
            # Normalize from [0, 255] to [0.0, 1.0]
            image = tf.decode_raw(image, tf.uint8)
            image = tf.cast(image, tf.float32)
            # image = tf.reshape(image, [784])
            return image / 255.0

        def decode_label(label):
            label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
            label = tf.reshape(label, [])  # label is a scalar
            return tf.to_int32(label)

        images = tf.data.FixedLengthRecordDataset(
            images_filepath, 28 * 28, header_bytes=16).map(decode_image)
        labels = tf.data.FixedLengthRecordDataset(
            labels_filepath, 1, header_bytes=8).map(decode_label)
        dataset = tf.data.Dataset.zip((images, labels))
        return dataset


    def download(directory, filename):
        """Download (and unzip) a file from the MNIST dataset if not already done."""
        filepath = os.path.join(directory, filename)
        if tf.gfile.Exists(filepath):
            return filepath
        if not tf.gfile.Exists(directory):
            tf.gfile.MakeDirs(directory)
        # CVDF mirror of http://yann.lecun.com/exdb/mnist/
        url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
        zipped_filepath = filepath + '.gz'
        print('Downloading %s to %s' % (url, zipped_filepath))
        urllib.request.urlretrieve(url, zipped_filepath)
        with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(zipped_filepath)
        return filepath



    def get_train_iterator(self, dataset):
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_oneshot_iterator()
        return iterator


    def get_train_iterator(self, dataset):
        TEST_NUM = 10000  # the number of mnist-testset
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(TEST_NUM)
        iterator = dataset.make_oneshot_iterator()
        return iterator


    def train_input_fn(self):
        dataset = self.get_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
        iterator = self.get_train_iterator(dataset)
        images, labels = iterator.get_next()
        return images, labels


    def test_input_fn(self):
        dataset = self.get_dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
        iterator = self.get_test_iterator(dataset)
        images, labels = iterator.get_next()
        return images, labels

    def predict_input_fn(self):
        raise NotImplementedError


