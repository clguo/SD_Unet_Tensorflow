import tensorflow as tf
from tensorflow.python.keras import backend as K


class DropBlock(tf.keras.layers.Layer):
    def __init__(self, keep_prob=1, block_size=7, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.keep_prob = keep_prob
        self.block_size = block_size

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.h = input_shape[1]
        self.w = input_shape[2]
        self.channel = input_shape[3]

        # pad the mask
        bottom = right = (self.block_size - 1) // 2
        top = left = (self.block_size - 1) - bottom
        self.padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
        self.set_keep_prob()
        super(DropBlock, self).build(input_shape)

    def call(self, inputs, training=None, scale=True, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (tf.to_float(self.block_size) ** 2) / ((w - tf.to_float(self.block_size) + 1) * (h - tf.to_float(self.block_size) + 1))

    def _create_mask(self, input_shape):

        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = DropBlock._bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, 1 ,1, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

    @staticmethod
    def _bernoulli(shape, mean):
        return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

