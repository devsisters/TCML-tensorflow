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
    return flags.parse_args()


class TCML:
    def __init__(self, hparams):
        self.num_classes = hparams.num_classes
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.input_dim = hparams.input_dim
        self.num_dense_filter = hparams.num_dense_filter
        self.dilation = hparams.dilation
        self.attention_value_dim = hparams.attension_value_dim

        self.filter_width = 2

        self.input_placeholder = tf.placeholder(tf.int32, [None, self.seq_len, self.input_dim])
        self.label_placeholder = tf.placeholder(tf.int32, [None, self.seq_len, ])

        self.dense_blocks = []
        for i, dilation in enumerate(self.dilation):
            with tf.variable_scope(f"dilation{i}({dilation})"):
                self.dense_blocks.append(self.generate_dense_block(self.input_placeholder, dilation))

    def _causal_conv(self, x, dilation, num_filter):
        with tf.variable_scope("causal_conv"):
            # input shape : [B, T, D]
            # filter_shape : spatial_filter_shape + [in_channels, out_channels]
            filter_shape = [self.filter_width, num_filter, num_filter]
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            filter = tf.get_variable("filter", shape=filter_shape, dtype=tf.float32,
                                     initializer=initializer)
            return tf.nn.convolution(x, filter,
                                     padding="SAME",
                                     dilation_rate=[1, dilation, 1])

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
