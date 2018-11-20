from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score as sklearn_f1
from tensorflow.python.framework import ops
from .. import f1_score


class TestF1Binary(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(1)

    @staticmethod
    def f1_score(*args, **kwargs):
        return f1_score(average='binary', num_classes=2, *args, **kwargs)

    def testVars(self):
        ops.reset_default_graph()
        self.f1_score(predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
        expected = ('f1_binary/precision/true_positives/count:0',
                    'f1_binary/recall/true_positives/count:0',
                    'f1_binary/precision/false_positives/count:0',
                    'f1_binary/recall/false_negatives/count:0')
        self.assertEqual(set(expected), set(v.name for v in tf.local_variables()))
        self.assertEqual(set(expected), set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))

    def testMetricsCollection(self):
        my_collection_name = '__metrics__'
        mean, _ = self.f1_score(
            predictions=tf.ones((10, 1)),
            labels=tf.ones((10, 1)),
            metrics_collections=[my_collection_name])
        self.assertListEqual(ops.get_collection(my_collection_name), [mean])

    def testUpdatesCollection(self):
        my_collection_name = '__updates__'
        _, update_op = self.f1_score(
            predictions=tf.ones((10, 1)),
            labels=tf.ones((10, 1)),
            updates_collections=[my_collection_name])
        self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

    def testValueTensorIsIdempotent(self):
        predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
        labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
        f1_value, update_op = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())

            # Run several updates.
            for _ in range(10):
                sess.run(update_op)

            # Then verify idempotency.
            initial_f1 = f1_value.eval()
            for _ in range(10):
                self.assertEqual(initial_f1, f1_value.eval())

    def testAllCorrect(self):
        inputs = np.random.randint(0, 2, size=(100, 1))

        predictions = tf.constant(inputs)
        labels = tf.constant(inputs)
        f1_value, update_op = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            self.assertAlmostEqual(1, sess.run(update_op))
            self.assertAlmostEqual(1, f1_value.eval())

    def testSomeCorrect_multipleInputDtypes(self):
        for dtype in (tf.bool, tf.int32, tf.float32):
            predictions = tf.cast(tf.constant([1, 0, 1, 0], shape=(1, 4)), dtype=dtype)
            labels = tf.cast(tf.constant([0, 1, 1, 0], shape=(1, 4)), dtype=dtype)
            f1_value, update_op = self.f1_score(labels, predictions)

            with self.test_session() as sess:
                sess.run(tf.local_variables_initializer())
                self.assertAlmostEqual(0.5, update_op.eval())
                self.assertAlmostEqual(0.5, f1_value.eval())

    def testWeighted1d(self):
        predictions = tf.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
        labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        f1_value, update_op = self.f1_score(labels, predictions, weights=tf.constant([[2], [5]]))

        with self.test_session():
            tf.local_variables_initializer().run()
            weighted_tp = 2.0 + 5.0
            weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
            expected_f1 = weighted_tp / weighted_positives
            self.assertAlmostEqual(expected_f1, update_op.eval())
            self.assertAlmostEqual(expected_f1, f1_value.eval())

    def testWeightedScalar_placeholders(self):
        predictions = tf.placeholder(dtype=tf.float32)
        labels = tf.placeholder(dtype=tf.float32)
        feed_dict = {
            predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
            labels: ((0, 1, 1, 0), (1, 0, 0, 1))
        }
        f1_value, update_op = self.f1_score(labels, predictions, weights=2)

        with self.test_session():
            tf.local_variables_initializer().run()
            weighted_tp = 2.0 + 2.0
            weighted_positives = (2.0 + 2.0) + (2.0 + 2.0)
            expected_f1 = weighted_tp / weighted_positives
            self.assertAlmostEqual(expected_f1, update_op.eval(feed_dict=feed_dict))
            self.assertAlmostEqual(expected_f1, f1_value.eval(feed_dict=feed_dict))

    def testWeighted1d_placeholders(self):
        predictions = tf.placeholder(dtype=tf.float32)
        labels = tf.placeholder(dtype=tf.float32)
        feed_dict = {
            predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
            labels: ((0, 1, 1, 0), (1, 0, 0, 1))
        }
        f1_value, update_op = self.f1_score(labels, predictions, weights=tf.constant([[2], [5]]))

        with self.test_session():
            tf.local_variables_initializer().run()
            weighted_tp = 2.0 + 5.0
            weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
            expected_f1 = weighted_tp / weighted_positives
            self.assertAlmostEqual(expected_f1, update_op.eval(feed_dict=feed_dict))
            self.assertAlmostEqual(expected_f1, f1_value.eval(feed_dict=feed_dict))

    def testWeighted2d(self):
        predictions = tf.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
        labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        f1_value, update_op = self.f1_score(
            labels,
            predictions,
            weights=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

        with self.test_session():
            tf.local_variables_initializer().run()
            weighted_tp = 3.0 + 4.0
            weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
            expected_f1 = weighted_tp / weighted_positives
            self.assertAlmostEqual(expected_f1, update_op.eval())
            self.assertAlmostEqual(expected_f1, f1_value.eval())

    def testWeighted2d_placeholders(self):
        predictions = tf.placeholder(dtype=tf.float32)
        labels = tf.placeholder(dtype=tf.float32)
        feed_dict = {
            predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
            labels: ((0, 1, 1, 0), (1, 0, 0, 1))
        }
        f1_value, update_op = self.f1_score(
            labels,
            predictions,
            weights=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

        with self.test_session():
            tf.local_variables_initializer().run()
            weighted_tp = 3.0 + 4.0
            weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
            expected_f1 = weighted_tp / weighted_positives
            self.assertAlmostEqual(expected_f1, update_op.eval(feed_dict=feed_dict))
            self.assertAlmostEqual(expected_f1, f1_value.eval(feed_dict=feed_dict))

    def testAllIncorrect(self):
        inputs = np.random.randint(0, 2, size=(100, 1))

        predictions = tf.constant(inputs)
        labels = tf.constant(1 - inputs)
        f1_value, update_op = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_op)
            self.assertAlmostEqual(0, f1_value.eval())

    def testZeroTrueAndFalsePositivesGivesZeroF1(self):
        predictions = tf.constant([0, 0, 0, 0])
        labels = tf.constant([0, 0, 0, 0])
        f1_value, update_op = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_op)
            self.assertEqual(0.0, f1_value.eval())

    def testAlmostAllFalse(self):
        predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        accuracy, update_accuracy = tf.metrics.accuracy(labels, predictions)
        f1_value, update_f1 = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run([update_accuracy, update_f1])
            self.assertAlmostEqual(0.9, accuracy.eval())
            self.assertEqual(0.0, f1_value.eval())

    def testAllTrue(self):
        predictions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        f1_value, update_f1 = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_f1)
            self.assertEqual(1.0, f1_value.eval())

    def testKnownResult(self):
        labels = [1, 0, 0, 1, 0, 1, 1, 1]
        predictions = [1, 1, 0, 0, 0, 1, 0, 1]

        known_value = sklearn_f1(labels, predictions, average='binary')
        f1_value, update_f1 = f1_score(labels, predictions, average='binary')

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_f1)
            self.assertAlmostEqual(known_value, f1_value.eval())


class TestF1Macro(TestF1Binary):
    @staticmethod
    def f1_score(*args, **kwargs):
        return f1_score(average='macro', num_classes=2, *args, **kwargs)

    def testVars(self):
        ops.reset_default_graph()
        self.f1_score(predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
        expected = ('f1_macro/precision_0/true_positives/count:0',
                    'f1_macro/precision_0/false_positives/count:0',
                    'f1_macro/precision_1/true_positives/count:0',
                    'f1_macro/precision_1/false_positives/count:0',
                    'f1_macro/recall_0/true_positives/count:0',
                    'f1_macro/recall_0/false_negatives/count:0',
                    'f1_macro/recall_1/true_positives/count:0',
                    'f1_macro/recall_1/false_negatives/count:0')
        self.assertEqual(set(expected), set(v.name for v in tf.local_variables()))
        self.assertEqual(set(expected), set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))

    def testAlmostAllFalse(self):
        predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        accuracy, update_accuracy = tf.metrics.accuracy(labels, predictions)
        f1_value, update_f1 = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run([update_accuracy, update_f1])
            self.assertAlmostEqual(0.9, accuracy.eval())
            self.assertAlmostEqual(0.4736842, f1_value.eval())

    def testZeroTrueAndFalsePositivesGivesZeroF1(self):
        predictions = tf.constant([0, 0, 0, 0])
        labels = tf.constant([0, 0, 0, 0])
        f1_value, update_op = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_op)
            self.assertEqual(0.5, f1_value.eval())

    def testKnownResult(self):
        labels = [0, 1, 2, 0, 1, 2]
        predictions = [0, 2, 1, 0, 0, 1]

        known_value = sklearn_f1(labels, predictions, average='macro')
        f1_value, update_f1 = f1_score(labels, predictions, average='macro', num_classes=3)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_f1)
            self.assertAlmostEqual(known_value, f1_value.eval())


class TestF1Micro(TestF1Binary):
    @staticmethod
    def f1_score(*args, **kwargs):
        return f1_score(average='micro', num_classes=2, *args, **kwargs)

    def testVars(self):
        ops.reset_default_graph()
        self.f1_score(predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
        expected = ('f1_micro/true_positives_0/count:0',
                    'f1_micro/true_positives_1/count:0',
                    'f1_micro/false_positives_0/count:0',
                    'f1_micro/false_positives_1/count:0',
                    'f1_micro/false_negatives_0/count:0',
                    'f1_micro/false_negatives_1/count:0')
        self.assertEqual(set(expected), set(v.name for v in tf.local_variables()))
        self.assertEqual(set(expected), set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))

    def testAlmostAllFalse(self):
        predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        accuracy, update_accuracy = tf.metrics.accuracy(labels, predictions)
        f1_value, update_f1 = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run([update_accuracy, update_f1])
            self.assertAlmostEqual(0.9, accuracy.eval())
            self.assertAlmostEqual(0.9, f1_value.eval())

    def testZeroTrueAndFalsePositivesGivesZeroF1(self):
        predictions = tf.constant([0, 0, 0, 0])
        labels = tf.constant([0, 0, 0, 0])
        f1_value, update_op = self.f1_score(labels, predictions)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_op)
            self.assertEqual(1.0, f1_value.eval())

    def testKnownResult(self):
        labels = [0, 1, 2, 0, 1, 2]
        predictions = [0, 2, 1, 0, 0, 1]

        known_value = sklearn_f1(labels, predictions, average='micro')
        f1_value, update_f1 = f1_score(labels, predictions, average='micro', num_classes=3)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(update_f1)
            self.assertAlmostEqual(known_value, f1_value.eval())
