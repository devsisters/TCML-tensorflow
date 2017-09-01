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
        self.q = tf.FIFOQueue(capacity, tf.float32, shapes=shapes)
        self.classes = classes
        self.inputs = inputs
        self.N = N
        self.K = K
        self.sess = sess

        self.coord = tf.train.Coordinator()

        self.placeholder = tf.placeholder(tf.float32, shapes)
        self.enqueue_op = self.q.enqueue(self.placeholder)

        self._run_enqueue_thread()

    def _make_one_data(self):
        """
        Extract K datas from N classes, concat and shuffle them.
        Then, add 1 data from random class(in N classes) at tail of data
        :return: (NK+1) x data
        """
        target_classes = random.sample(self.classes, self.N)
        dataset = []

        is_first = True
        last_data = None
        for class_name in target_classes:
            data_len = self.inputs[class_name].shape[0]
            if is_first:
                target_indices = np.random.choice(data_len, self.K + 1, False)
                target_datas = self.inputs[class_name][target_indices]
                target_datas, last_data, _ = np.split(target_datas, [self.K, self.K + 1])
                is_first = False
            else:
                target_indices = np.random.choice(data_len, self.K, False)
                target_datas = self.inputs[class_name][target_indices]
            dataset.append(target_datas)

        dataset_np = np.concatenate(dataset)
        np.random.shuffle(dataset_np)

        return np.append(dataset_np, last_data, axis=0)

    def _enqueue_thread_work(self):
        with self.coord.stop_on_exception():
            try:
                while not self.coord.should_stop():
                    new_data = self._make_one_data()
                    self.sess.run(self.enqueue_op, feed_dict={self.placeholder: new_data})
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
        q = FewShotInputQueue(10, inputs.keys(), inputs, [10, 2, 2], 3, 3, sess)

        with sess.as_default():
            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            # N.B. It's more efficient to reuse the same dequeue op in a loop.
            run_options = tf.RunOptions(timeout_in_ms=10000)
            result = sess.run(q.q.dequeue_many(5), options=run_options)
            print(result)


if __name__ == "__main__":
    _FewShotInputQueue_test()
