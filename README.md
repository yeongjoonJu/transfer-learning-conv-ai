# Pre-training for Multi-Turn Dialogue

This repo was cloned from [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai). This code is pre-training codes for multi-turn dialogue using GPT-2 or BART. We support standard Transformer model and uni-directional Transformer model such as `gpt2` and `facebook/bart-base`. Please check our data format in [here](https://wdprogrammer.tistory.com/90).

## Installation

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone https://github.com/yeongjoonJu/transfer-learning-conv-ai
cd transfer-learning-conv-ai
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Using the training script

The training script can be used in single GPU or multi GPU settings:

```bash
python pretrain_dialog.py  # Single GPU training
python -m torch.distributed.launch --nproc_per_node=3 pretrain_dialog.py [--eval_before_start]  # Training on 3 GPUs
```

## Using the interaction script

**--Not Yet--**

## Citation

If you use this code in your research, you can cite our NeurIPS CAI workshop [paper](http://arxiv.org/abs/1901.08149):

```bash
@article{DBLP:journals/corr/abs-1901-08149,
  author    = {Thomas Wolf and
               Victor Sanh and
               Julien Chaumond and
               Clement Delangue},
  title     = {TransferTransfo: {A} Transfer Learning Approach for Neural Network
               Based Conversational Agents},
  journal   = {CoRR},
  volume    = {abs/1901.08149},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.08149},
  archivePrefix = {arXiv},
  eprint    = {1901.08149},
  timestamp = {Sat, 02 Feb 2019 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-08149},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```