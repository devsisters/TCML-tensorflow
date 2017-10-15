# TCML-tensorflow
A tensorflow implementation of [Meta-Learning with Temporal Convolutions](https://arxiv.org/abs/1707.03141)

Embedding for Omniglot dataset is only available now.

# Prerequisites
* Python 3.6+
* Tensorflow 1.3
* Image dataset for training/validation (Python dictionary-like object)

# Usage
```
python train.py --dataset omniglot --n 5 --k 1 --dilation 1 2 1 2 1 2 1 2 4 8 16 --lr 5e-4 --batch_size 64
```

It will print valiation loss and accuracy.
Checkpoints and summaries for other metrics are saved in `./runs/tcml_{input_dim}_{num_dense_filter}_{attention_value_dim}_{lr}`

# License
MIT

# Author
Donghwa Kim ([@storykim](https://github.com/storykim))
