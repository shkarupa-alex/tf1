from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops


def f1_score(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    with tf.variable_scope(name, 'f1_score', (predictions, labels, weights)):
        precision_val, precision_upd = tf.metrics.precision(
            labels,
            predictions,
            weights,
            None,
            None,
            'precision'
        )
        recall_val, recall_upd = tf.metrics.recall(
            labels,
            predictions,
            weights,
            None,
            None,
            'recall'
        )

        def compute_f1_score(precision, recall, name):
            with tf.name_scope(name, 'compute', [precision, recall]):
                return tf.where(
                    tf.greater(precision + recall, 0),
                    2. * tf.div(
                        precision * recall,
                        precision + recall
                    ),
                    0,
                    name)

        value = compute_f1_score(precision_val, recall_val, 'value')
        update_op = compute_f1_score(precision_upd, recall_upd, 'update_op')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, value)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return value, update_op
