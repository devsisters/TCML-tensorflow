import numpy as np
import random


class FewShotInputQueue:
    def __init__(self, classes, inputs, N, K):
        """
        Initialize

        :param capacity: int. capacity of queue
        :param inputs: dict that key is class, value is data(d0, ..., dn)
        """
        self.classes = classes
        self.inputs = inputs
        self.N = N
        self.K = K

    def make_one_data(self):
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
               np.append(label_set, [last_class_idx], axis=0).astype(np.int32)


def _make_dummy_inputs():
    # 3 classes, 4 datas, 2x2 dim.
    input_dict = {}
    for i in range(3):
        input_dict[i] = np.multiply(np.ones([4, 2, 2], dtype=np.float32), i)

    return input_dict


def _FewShotInputQueue_test():
    inputs = _make_dummy_inputs()
    q = FewShotInputQueue(inputs.keys(), inputs, 3, 3)

    result, result_label = zip(*[q.make_one_data() for _ in range(10)])

    print(result, result_label)


if __name__ == "__main__":
    _FewShotInputQueue_test()
