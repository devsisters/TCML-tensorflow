import tensorflow as tf
import numpy as np
import argparse


def define_flags():
    flags = argparse.ArgumentParser()

    flags.add_argument("--num_classes", type=int, required=True, help="Number of classes[Required]")
    flags.add_argument("--batch_size", type=int, default=128, help="Batch size B[128]")
    flags.add_argument("--seq_len", type=int, default=20, help="Sequence length T[20]")
    flags.add_argument("--input_dim", type=int, default=512, help="Dimension of input D[512]")
    flags.add_argument("--num_dense_filter", type=int, default=128, help="# of filter in Dense block[128]")
    flags.add_argument("--dilation", type=int, nargs='+', help="List of dilation size")
    flags.add_argument("--attention_value_dim", type=int, default=16, help="Dimension of attension value d'[16]")
    flags.add_argument("--lr", type=float, default=1e-3, help="Learning rate[1e-3]")
    return flags.parse_args()


class TCML:
    def __init__(self, hparams, is_train):
        self.num_classes = hparams.num_classes
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.input_dim = hparams.input_dim
        self.num_dense_filter = hparams.num_dense_filter
        self.dilation = hparams.dilation
        self.attention_value_dim = hparams.attension_value_dim
        self.lr = hparams.lr

        self.filter_width = 2

        self.input_placeholder = tf.placeholder(tf.int32, [None, self.seq_len, self.input_dim])
        self.label_placeholder = tf.placeholder(tf.int32, [None, self.seq_len, ])
        self.is_train = is_train

        self.dense_blocks = []

        last_output = self.input_placeholder
        for i, dilation in enumerate(self.dilation):
            name = f"dilation{i}({dilation})"
            with tf.variable_scope(name):
                output = self.generate_dense_block(last_output, dilation)
                self.dense_blocks.append((name, output))

        # last_output : [B, T, D + 128 * i]
        _, T, d = last_output.get_shape().as_list()
        with tf.variable_scope("attention"):
            kernel_size = [1, d, self.attention_value_dim]  # width, in_channel, out_channel
            conv_kernel = tf.get_variable("1x1_conv", kernel_size,
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())

            attention_value = tf.nn.conv1d(last_output, conv_kernel, 1, "SAME")

            # dummy key & value at t=0
            self.key_t0 = key_t0 = tf.get_variable("key_t0", [T, d],
                                                   dtype=tf.float32,
                                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.value_t0 = tf.nn.conv1d(key_t0, conv_kernel, 1, "SAME")

            attention_outputs = self.attention_layer(last_output, attention_value, T, d)

        # attention_output : [B, T, d']
        # channel-wise softmax
        with tf.variable_scope("softmax"):
            kernel_size = [1, self.attention_value_dim, self.num_classes]
            conv_kernel = tf.get_variable("1x1_conv", kernel_size,
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
            softmax_vector = tf.nn.conv1d(attention_outputs, conv_kernel, 1, "SAME")

        self.loss = loss = tf.contrib.seq2seq.sequence_loss(softmax_vector,
                                                            self.label_placeholder,
                                                            tf.ones([self.batch_size, self.seq_len], dtype=tf.float32))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def _causal_conv(self, x, dilation, num_filter):
        with tf.variable_scope("causal_conv"):
            # input shape : [B, T, D]
            # filter_shape : spatial_filter_shape + [in_channels, out_channels]
            filter_shape = [self.filter_width, num_filter, num_filter]
            initializer = tf.contrib.layers.xavier_initializer_conv2d()

            tanh_filter = tf.get_variable("filter", shape=filter_shape, dtype=tf.float32,
                                          initializer=initializer)
            sigmoid_filter = tf.get_variable("filter", shape=filter_shape, dtype=tf.float32,
                                             initializer=initializer)

            x_reverse = tf.reverse(x, axis=2)

            tanh_output = tf.tanh(tf.nn.convolution(x_reverse, tanh_filter,
                                                    padding="SAME",
                                                    dilation_rate=[1, dilation, 1]))
            sigmoid_output = tf.sigmoid(tf.nn.convolution(x_reverse, sigmoid_filter,
                                                          padding="SAME",
                                                          dilation_rate=[1, dilation, 1]))

            return tf.reverse(tf.multiply(tanh_output, sigmoid_output), axis=2)

    def _residual_block(self, x, dilation, num_filter):
        # input shape : [B, T, D]
        with tf.variable_scope("residual_block"):
            # [filter_height, filter_width, in_channels, out_channels]
            conv_output = self._causal_conv(x, dilation, num_filter)
            return x + conv_output

    def generate_dense_block(self, x, dilation):
        # input shape : [B, T, D]
        conv = self._causal_conv(x, dilation, self.num_dense_filter)
        residual1 = self._residual_block(conv, dilation, self.num_dense_filter)
        residual2 = self._residual_block(residual1, dilation, self.num_dense_filter)
        return tf.concat([x, residual2], axis=2)

    def attention_layer(self, keys, values, T, d):
        # keys : B x T x d
        # value : B x T x d'
        # query : keys. B x T x d
        results = []  # list of ( B x d' )
        for i in range(T):
            # key : B x (t-1) x d
            if i == 0:
                # special case.
                key = self.key_t0  # 1 x d
                query = tf.gather(keys, i, axis=1)  # B x 1 x d
                value = self.value_t0  # 1 x d'

                attention = tf.nn.softmax(tf.divide(tf.einsum("ijk,lk->ijl", query, key), tf.sqrt(d)))  # B x 1 x (t-1)
                result = tf.einsum("ijk,kl->ijl", attention, value)  # B x d'
                results.append(result)
            else:
                key = tf.gather(keys, range(i), axis=1)
                query = tf.gather(keys, i, axis=1)
                value = tf.gather(values, range(i), axis=1)

                attention = tf.nn.softmax(tf.divide(tf.matmul(query, key, transpose_b=True), tf.sqrt(d)))  # 1 x (t-1)
                result = tf.matmul(attention, value)  # B x d'
                results.append(result)

        return tf.stack(results, axis=1)
