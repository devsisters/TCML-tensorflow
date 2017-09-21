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
        q = FewShotInputQueue(inputs.keys(), inputs, hparams.n, hparams.k)
        valid_q = FewShotInputQueue(valid_inputs.keys(), valid_inputs, hparams.n, hparams.k)

        generated_input, generated_label = tf.py_func(q.make_one_data, [], [tf.float32, tf.int32])
        batch_tensors = tf.train.batch([generated_input, generated_label], batch_size=hparams.batch_size, num_threads=4,
                                       shapes=[input_size, (episode_len,)])
        valid_input, valid_label = tf.py_func(valid_q.make_one_data, [], [tf.float32, tf.int32])
        valid_batch_tensors = tf.train.batch([valid_input, valid_label], batch_size=hparams.batch_size, num_threads=4,
                                             shapes=[input_size, (episode_len,)])

        with tf.variable_scope("networks"):
            embed_network = OmniglotEmbedNetwork(batch_tensors, hparams.batch_size)
            tcml = TCML(hparams, embed_network.output, embed_network.label_placeholder, True)

        with tf.variable_scope("networks", reuse=True):
            valid_embed_network = OmniglotEmbedNetwork(valid_batch_tensors, hparams.batch_size)
            valid_tcml = TCML(hparams, valid_embed_network.output, valid_embed_network.label_placeholder, False)

        params_to_str = f"tcml_{hparams.input_dim}_{hparams.num_dense_filter}_{hparams.attention_value_dim}_{hparams.lr}"
        log_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", params_to_str))

        # Summaries
        tf.summary.scalar("train_loss", tcml.loss)
        tf.summary.scalar("train_acc", tcml.accuracy)

        tf.summary.scalar("valid_loss", valid_tcml.loss)
        tf.summary.scalar("valid_acc", valid_tcml.accuracy)

        tf.summary.image("inputs", valid_embed_network.input_placeholder[0], max_outputs=episode_len)

        # Supervisor
        supervisor = tf.train.Supervisor(
            logdir=log_dir,
            save_summaries_secs=120,
            save_model_secs=600,
            global_step=tcml.global_step,
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print("Training start")

        with supervisor.managed_session(config=config) as sess:
            min_dev_loss = 10000
            min_step = -1

            STEP_NUM = 10000000
            EARLY_STOP = 30000000
            print_every = 500

            HUGE_VALIDATION_CYCLE = print_every * 20

            last_dev = time.time()

            for step in range(STEP_NUM):
                if supervisor.should_stop():
                    break

                if step - min_step > EARLY_STOP:
                    print("Early stopping...")
                    break

                if step % print_every != 0:
                    _, loss, acc, global_step = sess.run(
                        [tcml.train_step, tcml.loss, tcml.accuracy, tcml.global_step])
                else:
                    _, loss, acc, global_step = sess.run(
                        [tcml.train_step, tcml.loss, tcml.accuracy, tcml.global_step])

                    loss, acc = sess.run([valid_tcml.loss, valid_tcml.accuracy])

                    current_time = time.time()
                    print(
                        f'Evaluate(Step {step}/{global_step} : valid loss({loss}), acc({acc}) in {current_time - last_dev} s')

                    # HUGE VALIDATION
                    if step != 0 and step % HUGE_VALIDATION_CYCLE == 0:
                        total_loss = total_acc = 0.
                        BATCH_NUM = 30
                        for _ in range(BATCH_NUM):
                            loss, acc = sess.run([valid_tcml.loss, valid_tcml.accuracy])
                            total_loss += loss * hparams.batch_size
                            total_acc += acc * hparams.batch_size

                        total_loss /= BATCH_NUM * hparams.batch_size
                        total_acc /= BATCH_NUM * hparams.batch_size

                        huge_data_acc_summary = tf.Summary()
                        huge_data_acc_summary.value.add(tag="huge_data_accuracy", simple_value=total_acc)
                        supervisor.summary_computed(sess, huge_data_acc_summary, global_step=global_step)

                        huge_data_loss_summary = tf.Summary()
                        huge_data_loss_summary.value.add(tag="huge_data_loss", simple_value=total_loss)
                        supervisor.summary_computed(sess, huge_data_loss_summary, global_step=global_step)

                    last_dev = current_time

                    if loss < min_dev_loss:
                        min_dev_loss = loss
                        min_step = step


if __name__ == "__main__":
    train()
