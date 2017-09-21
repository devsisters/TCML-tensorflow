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
        valid_path = "data/omniglot/test.npz"
    else:
        raise NotImplementedError

    if hparams.dataset == "omniglot":
        input_size = (episode_len, 28, 28, 1)
    else:
        raise NotImplementedError

    with open(input_path, "rb") as f:
        input_npz = np.load(f)
        inputs = {}
        for filename in input_npz.files:
            inputs[filename] = input_npz[filename]

    with open(valid_path, "rb") as f:
        valid_npz = np.load(f)
        valid_inputs = {}
        for filename in valid_npz.files:
            valid_inputs[filename] = valid_npz[filename]

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        q = FewShotInputQueue(inputs.keys(), inputs, hparams.n, hparams.k)
        valid_q = FewShotInputQueue(valid_inputs.keys(), valid_inputs, hparams.n, hparams.k)

        generated_input, generated_label = tf.py_func(q.make_one_data, [], [tf.float32, tf.int32])
        batch_tensors = tf.train.batch([generated_input, generated_label], batch_size=hparams.batch_size, num_threads=4,
                                       shapes=[input_size, (episode_len,)])
        a, b = tf.py_func(valid_q.make_one_data, [], [tf.float32, tf.int32])
        valid_batch_tensors = tf.train.batch([a, b], batch_size=hparams.batch_size, num_threads=4,
                                             shapes=[input_size, (episode_len,)])

        with tf.variable_scope("networks"):
            embed_network = OmniglotEmbedNetwork(batch_tensors, hparams.batch_size)
            tcml = TCML(hparams, embed_network.output, embed_network.label_placeholder, True)

        with tf.variable_scope("networks", reuse=True):
            valid_embed_network = OmniglotEmbedNetwork(valid_batch_tensors, hparams.batch_size)
            valid_tcml = TCML(hparams, valid_embed_network.output, valid_embed_network.label_placeholder, True)

        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        tcml.global_step = global_step

        params_to_str = f"tcml_{hparams.input_dim}_{hparams.num_dense_filter}_{hparams.attention_value_dim}_{hparams.lr}"
        log_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", params_to_str))

        train_loss_summary = tf.summary.scalar("train_loss", tcml.loss)
        train_acc_summary = tf.summary.scalar("train_acc", tcml.accuracy)

        train_merged = tf.summary.merge([train_loss_summary, train_acc_summary])

        valid_loss_summary = tf.summary.scalar("valid_loss", tcml.loss)
        valid_acc_summary = tf.summary.scalar("valid_acc", tcml.accuracy)

        valid_merged = tf.summary.merge([valid_loss_summary, valid_acc_summary])

        input_summary = tf.summary.image("inputs", valid_embed_network.input_placeholder[0], max_outputs=episode_len)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        
        print("Training start")

        with sess:
            tf.train.start_queue_runners()
            min_dev_loss = 10000
            min_step = -1

            STEP_NUM = 10000000
            EARLY_STOP = 30000000
            print_every = 500
            last_dev = time.time()

            sess.run(tf.initialize_all_variables())

            for step in range(STEP_NUM):
                if step - min_step > EARLY_STOP:
                    print("Early stopping...")
                    break

                if step % print_every != 0:
                    _, loss, acc = sess.run(
                        [tcml.train_step, tcml.loss, tcml.accuracy])
                else:
                    _, loss, acc, train_summary = sess.run(
                        [tcml.train_step, tcml.loss, tcml.accuracy, train_merged])

                    loss, acc, valid_summary, input_summary_eval = sess.run([valid_tcml.loss, valid_tcml.accuracy, valid_merged, input_summary])
                    summary_writer.add_summary(train_summary, step)
                    summary_writer.add_summary(valid_summary, step)
                    summary_writer.add_summary(input_summary_eval, step)

                    current_time = time.time()
                    print(
                        f'Evaluate(Step {step} : valid loss({loss}), acc({acc}) in {current_time - last_dev} s')
                    last_dev = current_time

                    if loss < min_dev_loss:
                        min_dev_loss = loss
                        min_step = step


if __name__ == "__main__":
    train()
