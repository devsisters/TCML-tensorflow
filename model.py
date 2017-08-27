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
    return flags.parse_args()


class TCML:
    def __init__(self, hparams):
        pass
