import tensorflow as tf
import numpy as np


class OmniglotEmbedNetwork:
    def __init__(self, inputs, batch_size):
        '''
         4 blocks of
           {3 × 3 conv (64 filters),
           batch normalization,
           leaky ReLU activation (leak 0.1),
           and 2 × 2 max-pooling}
        '''

        self.epsilon = 1e-10

        # input : B x T x H x W x C
        # output : B x T x D
        self.input_placeholder, self.label_placeholder = inputs

        with tf.variable_scope("omni_embed_0"):
            last_output = self.add_block(self.input_placeholder, 1, 64)

        for i in [1, 2, 3]:
            with tf.variable_scope(f"omni_embed_{i}"):
                last_output = self.add_block(last_output, 64, 64)

        self.output = tf.squeeze(last_output)

    def add_block(self, x, in_channel, out_channel):
        kernel_size = [1, 3, 3, in_channel, out_channel]
        kernel = tf.get_variable("kernel", kernel_size, dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_output = tf.nn.conv3d(x, kernel, [1, 1, 1, 1, 1], "SAME")

        beta = tf.get_variable('beta', [out_channel], initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', [out_channel], initializer=tf.constant_initializer(1.0))

        batch_mean, batch_var = tf.nn.moments(conv_output, [0, 1, 2, 3])
        batch_normalized = tf.nn.batch_normalization(conv_output, batch_mean, batch_var, beta, gamma, self.epsilon)

        relu_output = tf.nn.relu(batch_normalized) - 0.1 * tf.nn.relu(-batch_normalized)

        return tf.nn.max_pool3d(relu_output, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], "VALID")


def _OmniglotEmbed_test():
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        dummy_input = np.random.rand(10, 28, 28, 1)
        dummy_label = np.random.randint(5, size=(10, ))
        queue = tf.RandomShuffleQueue(20,
                                      min_after_dequeue=2,
                                      shapes=[dummy_input.shape, dummy_label.shape], dtypes=[tf.float32, tf.int32])
        enqueue = queue.enqueue([dummy_input, dummy_label])
        qr = tf.train.QueueRunner(queue, [enqueue] * 2)
        tf.train.add_queue_runner(qr)

        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

        model = OmniglotEmbedNetwork(queue, 5)

        with sess.as_default():
            init = tf.initialize_all_variables()
            sess.run(init)

            output, = sess.run([model.output])
            print(output.shape)

            coord.request_stop()
            coord.join(enqueue_threads)

if __name__ == "__main__":
    _OmniglotEmbed_test()
