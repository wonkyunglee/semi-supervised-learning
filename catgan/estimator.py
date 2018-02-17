import tensorflow as tf

class CatGAN(object):

    def __init__(self, config):
        model_dir = config.model_dir



    def get_model_fn(self):

        network = self.network
        loss_fn = self.loss_fn
        pred_fn = self.pred_fn
        accuracy_fn = self.accuracy_fn
        learning_rate = self.learning_rate


        def model_fn(features, labels, mode, params):
            input_tensor = features
            logit = network(input_tensor)
            loss = loss_fn(logit, labels)
            preds = pred_fn(logit)
            accuracy = accuracy_fn(labels, preds)
            metrics = {'accuracy': accuracy}
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            # assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        return model_fn



    def get_estimator(self):

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn,
            params=self.params)
