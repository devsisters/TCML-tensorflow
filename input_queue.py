import tensorflow as tf
import numpy as np
import random
import threading


class FewShotInputQueue:
    def __init__(self, capacity, classes, inputs, shapes, N, K, sess):
        """
        Initialize

        :param capacity: int. capacity of queue
        :param inputs: dict that key is class, value is data(d0, ..., dn)
        """
        self.input_q = tf.FIFOQueue(capacity, [tf.float32, tf.int32], shapes=[shapes, shapes[:1]])
        self.classes = classes
        self.inputs = inputs
        self.N = N
        self.K = K
        self.sess = sess

        self.coord = tf.train.Coordinator()

        with tf.variable_scope("queue_placeholder"):
            self.input_placeholder = tf.placeholder(tf.float32, shapes)
            self.label_placeholder = tf.placeholder(tf.int32, shapes[:1])
            self.input_enqueue_op = self.input_q.enqueue([self.input_placeholder, self.label_placeholder])

        self._run_enqueue_thread()

    def _make_one_data(self):
        """
        Extract K datas from N classes, concat and shuffle them.
        Then, add 1 data from random class(in N classes) at tail of data
        :return: (NK+1) x data
        """
        target_classes = random.sample(self.classes, self.N)
        dataset = []
        label_set = []

        last_data = None

        last_class = random.sample(target_classes, 1)[0]
        last_class_idx = None
        for i, class_name in enumerate(target_classes):
            data_len = self.inputs[class_name].shape[0]
            if class_name == last_class:
                last_class_idx = i
                target_indices = np.random.choice(data_len, self.K + 1, False)
                target_datas = self.inputs[class_name][target_indices]
                target_datas, last_data, _ = np.split(target_datas, [self.K, self.K + 1])
            else:
                target_indices = np.random.choice(data_len, self.K, False)
                target_datas = self.inputs[class_name][target_indices]
            dataset.append(target_datas)
            label_set += [i] * self.K

        dataset_np = np.concatenate(dataset)
        perm = np.random.permutation(self.N * self.K)

        dataset_np = dataset_np[perm]
        label_set = np.asarray(label_set, np.int32)[perm]

        return np.expand_dims(np.append(dataset_np, last_data, axis=0), -1), \
               np.append(label_set, [last_class_idx], axis=0)

    def _enqueue_thread_work(self):
        with self.coord.stop_on_exception():
            try:
                while not self.coord.should_stop():
                    new_data, new_label = self._make_one_data()
                    # XXX: is it safe?
                    self.sess.run([self.input_enqueue_op],
                                  feed_dict={self.input_placeholder: new_data,
                                             self.label_placeholder: new_label})
            except Exception as e:
                print(e)

    def _run_enqueue_thread(self):
        self.threads = threads = [threading.Thread(target=self._enqueue_thread_work, daemon=True) for _ in range(4)]
        for thread in threads:
            thread.start()
            # map(lambda x: x.start(), threads)


def _make_dummy_inputs():
    # 3 classes, 4 datas, 2x2 dim.
    input_dict = {}
    for i in range(3):
        input_dict[i] = np.multiply(np.ones([4, 2, 2], dtype=np.float32), i)

    return input_dict


def _FewShotInputQueue_test():
    inputs = _make_dummy_inputs()
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        q = FewShotInputQueue(10, inputs.keys(), inputs, [10, 2, 2, 1], 3, 3, sess)

        with sess.as_default():
            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            # N.B. It's more efficient to reuse the same dequeue op in a loop.
            run_options = tf.RunOptions(timeout_in_ms=10000)
            result, result_label = sess.run(q.input_q.dequeue_many(10), options=run_options)
            print(result, result_label)


if __name__ == "__main__":
    _FewShotInputQueue_test()
