import tensorflow as tf
import numpy as np
import argparse


def define_flags():
    flags = argparse.ArgumentParser()

    flags.add_argument("--num_classes", type=int, default=None, help="Number of classes[Required]")
    flags.add_argument("--dilation", type=int, nargs='+', help="List of dilation size[Required]")

    flags.add_argument("--batch_size", type=int, default=128, help="Batch size B[128]")
    flags.add_argument("--seq_len", type=int, default=20, help="Sequence length T[20]")
    flags.add_argument("--input_dim", type=int, default=512, help="Dimension of input D[512]")
    flags.add_argument("--num_dense_filter", type=int, default=128, help="# of filter in Dense block[128]")
    flags.add_argument("--attention_value_dim", type=int, default=16, help="Dimension of attension value d'[16]")
    flags.add_argument("--lr", type=float, default=1e-3, help="Learning rate[1e-3]")
    return flags.parse_args()


class TCML:
    def __init__(self, hparams, is_train):
        assert hparams.num_classes is not None
        assert hparams.dilation is not None
        self.num_classes = hparams.num_classes
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.input_dim = hparams.input_dim
        self.num_dense_filter = hparams.num_dense_filter
        self.dilation = hparams.dilation
        self.attention_value_dim = hparams.attention_value_dim
        self.lr = hparams.lr

        self.filter_width = 2

        self.input_placeholder = tf.placeholder(tf.float32, [None, self.seq_len, self.input_dim])
        self.label_placeholder = tf.placeholder(tf.int32, [None, self.seq_len, ])
        self.is_train = is_train

        self.dense_blocks = []

        feed_label, target_label = tf.split(self.label_placeholder, [self.seq_len-1, 1],
                                            axis=1)
        feed_label_one_hot = tf.one_hot(feed_label,
                                        depth=self.num_classes,
                                        dtype=tf.float32)
        feed_label_one_hot = tf.concat([feed_label_one_hot, tf.zeros((self.batch_size, 1, self.num_classes))], axis=1)
        concated_input = tf.concat([self.input_placeholder, feed_label_one_hot], axis=2)

        last_output = concated_input
        d = self.input_dim + self.num_classes
        for i, dilation in enumerate(self.dilation):
            name = f"dilation{i}_{dilation}"
            with tf.variable_scope(name):
                last_output = output = self.generate_dense_block(last_output, d, dilation)
                self.dense_blocks.append((name, output))
                d += self.num_dense_filter

        # last_output : [B, T, D + 128 * i]
        with tf.variable_scope("attention"):
            kernel_size = [1, d, self.attention_value_dim]  # width, in_channel, out_channel
            conv_kernel = tf.get_variable("1x1_conv", kernel_size,
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())

            key, query = tf.split(last_output, [self.seq_len - 1, 1], axis=1)
            attention_value = tf.nn.conv1d(key, conv_kernel, 1, "SAME")
            attention_outputs = self.attention_layer(key, attention_value, query, float(d))

        # attention_output : [B, 1, d']
        # channel-wise softmax
        with tf.variable_scope("softmax"):
            kernel_size = [1, self.attention_value_dim, self.num_classes]
            conv_kernel = tf.get_variable("1x1_conv", kernel_size,
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
            softmax_vector = tf.nn.conv1d(attention_outputs, conv_kernel, 1, "SAME")

        self.loss = loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_label,
                                                                                         logits=softmax_vector))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def _causal_conv(self, x, dilation, in_channel, out_channel):
        with tf.variable_scope("causal_conv"):
            # input shape : [B, T, D]
            # filter_shape : spatial_filter_shape + [in_channels, out_channels]
            filter_shape = [self.filter_width, in_channel, out_channel]
            initializer = tf.contrib.layers.xavier_initializer_conv2d()

            tanh_filter = tf.get_variable("tanh_filter", shape=filter_shape, dtype=tf.float32,
                                          initializer=initializer)
            sigmoid_filter = tf.get_variable("sigmoid_filter", shape=filter_shape, dtype=tf.float32,
                                             initializer=initializer)

            x_reverse = tf.reverse(x, axis=[2])

            tanh_output = tf.tanh(tf.nn.convolution(x_reverse, tanh_filter,
                                                    padding="SAME",
                                                    dilation_rate=(dilation,)))
            sigmoid_output = tf.sigmoid(tf.nn.convolution(x_reverse, sigmoid_filter,
                                                          padding="SAME",
                                                          dilation_rate=(dilation,)))

            return tf.reverse(tf.multiply(tanh_output, sigmoid_output), axis=[2])

    def _residual_block(self, x, dilation, num_filter):
        # input shape : [B, T, D]
        # [filter_height, filter_width, in_channels, out_channels]
        conv_output = self._causal_conv(x, dilation, num_filter, num_filter)
        return x + conv_output

    def generate_dense_block(self, x, input_dim, dilation):
        # input shape : [B, T, D]
        conv = self._causal_conv(x, dilation, input_dim, self.num_dense_filter)
        with tf.variable_scope("residual_block_1"):
            residual1 = self._residual_block(conv, dilation, self.num_dense_filter)
        with tf.variable_scope("residual_block_2"):
            residual2 = self._residual_block(residual1, dilation, self.num_dense_filter)
        return tf.concat([x, residual2], axis=2)

    def attention_layer(self, key, value, query, d):
        # key : B x T-1 x d
        # value : B x T-1 x d'
        # query : B x 1 x d
        attention = tf.nn.softmax(tf.divide(tf.matmul(query, key, transpose_b=True), tf.sqrt(d)))  # 1 x (t-1)
        return tf.matmul(attention, value)  # B x d'


def _make_dummy_data():
    # 4 x 20 x 10 input data (float32)
    # 4 x 20 label data (int, [0, 4])
    input_data = np.random.randn(4, 20, 10)
    label_data = np.random.randint(5, size=(4, 20))
    return input_data, label_data


def _TCML_test():
    hparams = define_flags()
    hparams.num_classes = 5
    hparams.input_dim = 10
    hparams.num_dense_filter = 16
    hparams.batch_size = 4
    hparams.dilation = [1, 2, 1, 2]

    with tf.Graph().as_default():
        model = TCML(hparams, True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        with sess.as_default():
            init = tf.initialize_all_variables()
            sess.run(init)

            dummy_input, dummy_label = _make_dummy_data()

            feed_dict = {
                model.input_placeholder: dummy_input,
                model.label_placeholder: dummy_label,
            }

            _, loss = sess.run([model.train_step, model.loss], feed_dict=feed_dict)
            print(loss)


if __name__ == "__main__":
    _TCML_test()
