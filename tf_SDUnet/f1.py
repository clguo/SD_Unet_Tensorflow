from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops.metrics_impl import _remove_squeezable_dimensions, _aggregate_across_towers


def f1_score(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None,
             average='binary', num_classes=2):
    if context.executing_eagerly():
        raise RuntimeError('tf1.f1_score is not supported when eager execution is enabled.')

    if num_classes < 2:
        raise ValueError('Wrong F1 classes number: {}'.format(num_classes))

    if num_classes == 2 and average != 'binary':
        tf.logging.warning('Consider using "binary" average for F1-score with 2 classes.')

    if average == 'binary':
        return f1_binary(
            labels=labels,
            predictions=predictions,
            weights=weights,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections,
            name=name
        )
    elif average == 'macro':
        return f1_macro(
            labels=labels,
            predictions=predictions,
            num_classes=num_classes,
            weights=weights,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections,
            name=name
        )
    elif average == 'micro':
        return f1_micro(
            labels=labels,
            predictions=predictions,
            num_classes=num_classes,
            weights=weights,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections,
            name=name
        )
    else:
        raise ValueError('Wrong F1 average: {}'.format(average))


def f1_binary(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):

    print(labels.shape)
    if context.executing_eagerly():
        raise RuntimeError('tf1.f1_binary is not supported when eager execution is enabled.')

    with tf.variable_scope(name, 'f1_binary', (predictions, labels, weights)):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=tf.cast(predictions, dtype=tf.bool),
            labels=tf.cast(labels, dtype=tf.bool),
            weights=weights
        )
        print(labels.shape)
        print(labels)
        precision_val, precision_upd = tf.metrics.precision(
            labels=labels,
            predictions=predictions,
            weights=weights,
            metrics_collections=None,
            updates_collections=None,
            name='precision',
        )
        print(labels.shape)
        print(labels)
        recall_val, recall_upd = tf.metrics.recall(
            labels=labels,
            predictions=predictions,
            weights=weights,
            metrics_collections=None,
            updates_collections=None,
            name='recall'
        )
        print(labels.shape)
        print(labels)
        def compute_f1_binary(_precision, _recall, _name):
            return 2. * tf.div_no_nan(
                _precision * _recall,
                _precision + _recall,
                name=_name
            )

        def once_across_towers(_, _precision, _recall):
            return compute_f1_binary(_precision, _recall, 'value')

        value = _aggregate_across_towers(metrics_collections, once_across_towers, precision_val, recall_val)
        update_op = compute_f1_binary(precision_upd, recall_upd, 'update_op')

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        print(labels.shape)
        return value, update_op


def _select_class(labels, predictions, class_id):
    class_fill = tf.fill(tf.shape(labels), class_id)
    zeros_fill = tf.zeros_like(labels, dtype=tf.bool)
    ones_fill = tf.ones_like(labels, dtype=tf.bool)

    class_labels = tf.where(tf.equal(labels, class_fill), ones_fill, zeros_fill)
    class_predictions = tf.where(tf.equal(predictions, class_fill), ones_fill, zeros_fill)

    return class_labels, class_predictions


def f1_macro(labels, predictions, num_classes, weights=None, metrics_collections=None, updates_collections=None,
             name=None):
    if context.executing_eagerly():
        raise RuntimeError('tf1.f1_macro is not supported when eager execution is enabled.')

    with tf.variable_scope(name, 'f1_macro', (predictions, labels, weights)):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=tf.cast(predictions, dtype=tf.int32),
            labels=tf.cast(labels, dtype=tf.int32),
            weights=weights
        )

        precisions, recalls = [], []
        for class_id in range(num_classes):
            class_labels, class_predictions = _select_class(
                labels=labels,
                predictions=predictions,
                class_id=class_id
            )
            precisions.append(tf.metrics.precision(
                labels=class_labels,
                predictions=class_predictions,
                weights=weights,
                metrics_collections=None,
                updates_collections=None,
                name='precision_{}'.format(class_id),
            ))
            recalls.append(tf.metrics.recall(
                labels=class_labels,
                predictions=class_predictions,
                weights=weights,
                metrics_collections=None,
                updates_collections=None,
                name='recall_{}'.format(class_id),
            ))

        def compute_f1_macro(_precisions, _recalls, _name):
            _precision = tf.div(
                tf.add_n(_precisions),
                num_classes
            )
            _recall = tf.div(
                tf.add_n(_recalls),
                num_classes
            )

            return 2. * tf.div_no_nan(
                _precision * _recall,
                _precision + _recall,
                name=_name
            )

        def once_across_towers(_, _precisions, _recalls):
            return compute_f1_macro(_precisions, _recalls, 'value')

        value = _aggregate_across_towers(
            metrics_collections,
            once_across_towers,
            [p for p, _ in precisions],
            [r for r, _ in recalls]
        )
        update_op = compute_f1_macro(
            [p for _, p in precisions],
            [r for _, r in recalls],
            'update_op'
        )

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return value, update_op


def f1_micro(labels, predictions, num_classes, weights=None, metrics_collections=None, updates_collections=None,
             name=None):
    if context.executing_eagerly():
        raise RuntimeError('tf1.f1_micro is not supported when eager execution is enabled.')

    with tf.variable_scope(name, 'f1_micro', (predictions, labels, weights)):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=tf.cast(predictions, dtype=tf.int32),
            labels=tf.cast(labels, dtype=tf.int32),
            weights=weights
        )

        tps, fps, fns = [], [], []
        for class_id in range(num_classes):
            class_labels, class_predictions = _select_class(
                labels=labels,
                predictions=predictions,
                class_id=class_id
            )
            tps.append(tf.metrics.true_positives(
                labels=class_labels,
                predictions=class_predictions,
                weights=weights,
                metrics_collections=None,
                updates_collections=None,
                name='true_positives_{}'.format(class_id),
            ))
            fps.append(tf.metrics.false_positives(
                labels=class_labels,
                predictions=class_predictions,
                weights=weights,
                metrics_collections=None,
                updates_collections=None,
                name='false_positives_{}'.format(class_id),
            ))
            fns.append(tf.metrics.false_negatives(
                labels=class_labels,
                predictions=class_predictions,
                weights=weights,
                metrics_collections=None,
                updates_collections=None,
                name='false_negatives_{}'.format(class_id),
            ))

        def compute_f1_micro(_tps, _fps, _fns, _name):
            _precision = tf.div_no_nan(
                tf.add_n(_tps),
                tf.add_n(_tps + _fps),
            )
            _recall = tf.div_no_nan(
                tf.add_n(_tps),
                tf.add_n(_tps + _fns),
            )

            return 2. * tf.div_no_nan(
                _precision * _recall,
                _precision + _recall,
                name=_name
            )

        def once_across_towers(_, _tps, _fps, _fns):
            return compute_f1_micro(_tps, _fps, _fns, 'value')

        value = _aggregate_across_towers(
            metrics_collections,
            once_across_towers,
            [tp for tp, _ in tps],
            [fp for fp, _ in fps],
            [fn for fn, _ in fns],
        )
        update_op = compute_f1_micro(
            [tp for _, tp in tps],
            [fp for _, fp in fps],
            [fn for _, fn in fns],
            'update_op'
        )

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return value, update_op