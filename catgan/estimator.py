import tensorflow as tf

class Classifier(object):

    def __init__(self, config):
        self.params = config.params
        self.model_dir = config.model_dir

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):
            #x = tf.feature_column.numeric_column('x')
            #columns = [x]
            #input_tensor = tf.feature_column.input_layer(features, columns)
            input_tensor = features['x']
            logits = network(input_tensor, params)
            loss = loss_fn(logits, labels)
            preds = pred_fn(logits)
            accuracy, acc_update_op = accuracy_fn(labels, preds)
            metrics = {'accuracy': (accuracy, acc_update_op)}
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            # assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            logging_hook = tf.train.LoggingTensorHook({'loss':loss, 'accuracy':accuracy},
                                                      every_n_iter=100)
            train_ops = tf.group(train_op, acc_update_op)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops,
                                              predictions=preds,
                                              training_hooks=[logging_hook])



        def loss_fn(logits, labels):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            return loss

        def pred_fn(logits):
            pred = tf.argmax(logits, 1)
            return pred

        def accuracy_fn(labels, preds):
            accuracy, update_op = tf.metrics.accuracy(labels=labels,
                                           predictions=preds,
                                           name='accuracy')
            return accuracy, update_op

        def network(input_tensor, params):
            fc1 = tf.layers.dense(inputs=input_tensor,
                                  units=params['hidden_units'][0],
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.random_normal(0.0, 0.01))
            fc2 = tf.layers.dense(inputs=fc1,
                                  units=params['hidden_units'][1],
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.random_normal(0.0, 0.01))
            logits = tf.layers.dense(inputs=fc2,
                                  units=params['n_classes'],
                                  activation=None,
                                  kernel_initializer=tf.initializers.random_normal(0.0, 0.01))
            return logits

        return model_fn

    def get_estimator(self):

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.params,
            model_dir=self.model_dir)
        return estimator
