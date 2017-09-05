import argparse
import tensorflow as tf
import numpy as np
import os
import time
from model import TCML
from omniglot_embed import OmniglotEmbedNetwork
from input_queue import FewShotInputQueue


def define_flags():
    flags = argparse.ArgumentParser()

    flags.add_argument("--num_classes", type=int, default=None, help="Number of classes[Required]")
    flags.add_argument("--n", type=int, default=None, help="N [Required]")
    flags.add_argument("--k", type=int, default=None, help="K [Required]")
    flags.add_argument("--dataset", type=str, default="omniglot", help="Dataset (omniglot / miniimage) [omniglot]")

    flags.add_argument("--dilation", type=int, nargs='+', help="List of dilation size[Required]")

    flags.add_argument("--batch_size", type=int, default=128, help="Batch size B[128]")
    flags.add_argument("--input_dim", type=int, default=64, help="Dimension of input D[64]")
    flags.add_argument("--num_dense_filter", type=int, default=128, help="# of filter in Dense block[128]")
    flags.add_argument("--attention_value_dim", type=int, default=16, help="Dimension of attension value d'[16]")
    flags.add_argument("--lr", type=float, default=1e-3, help="Learning rate[1e-3]")
    return flags.parse_args()


def train():
    hparams = define_flags()
    hparams.seq_len = episode_len = hparams.n * hparams.k + 1

    if hparams.dataset == "omniglot":
        input_path = "data/omniglot/train.npz"
    else:
        raise NotImplementedError

    if hparams.dataset == "omniglot":
        input_size = (episode_len, 28, 28, 1)
    else:
        raise NotImplementedError

    f = open(input_path, "rb")
    inputs = np.load(f)

    with tf.Graph().as_default():
        sess = tf.Session()
        q = FewShotInputQueue(5 * episode_len, inputs.files, inputs, input_size, hparams.n, hparams.k, sess)
        embed_network = OmniglotEmbedNetwork(q.input_q, q.label_q, hparams.batch_size)
        tcml = TCML(hparams, embed_network.output, embed_network.label_placeholder, True)

        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        tcml.global_step = global_step

        params_to_str = f"tcml_{hparams.input_dim}_{hparams.num_dense_filter}_{hparams.attention_value_dim}_{hparams.lr}"
        log_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", params_to_str))

        train_loss_summary = tf.summary.scalar("train_loss", tcml.loss)
        train_acc_summary = tf.summary.scalar("train_acc", tcml.accuracy)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print("Training start")

        with sess.as_default():
            min_dev_loss = 10000
            min_step = -1

            STEP_NUM = 1000000
            EARLY_STOP = 3000
            print_every = 500
            last_dev = time.time()

            sess.run(tf.initialize_all_variables())

            for step in range(STEP_NUM):
                if step - min_step > EARLY_STOP:
                    print("Early stopping...")
                    break

                _, global_step, loss, acc = sess.run(
                    [tcml.train_step, tcml.global_step, tcml.loss, tcml.accuracy])

                if step % print_every == 0:
                    current_time = time.time()
                    print(
                        f'Evaluate(Step {step}/{global_step} : train loss({loss}), acc({acc}) in {current_time - last_dev} s')
                    last_dev = current_time

                if loss < min_dev_loss:
                    min_dev_loss = loss
                    min_step = step


if __name__ == "__main__":
    train()
