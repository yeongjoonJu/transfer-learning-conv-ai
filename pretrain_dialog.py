# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.metrics.nlp import Bleu
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, BartForConditionalGeneration, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, BartTokenizer, BartModel,
                                  GPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

import re
from utils import make_logdir
from dataloader import get_data_loaders

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def add_special_tokens_(model, tokenizer, attr_to_special_token):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(attr_to_special_token) # doesn't add if they are already there
    if num_added_tokens > 0:
        a = model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def decode_logit_label(logits, label, tokenizer):
    logits = torch.argmax(logits,dim=-1)
    y_pred = []
    y = []
    for b in range(label.shape[0]):
        target = label[b].tolist()
        pred = logits[b].tolist()
        bos_pos = target.index(tokenizer.bos_token_id)
        try:
            eos_pos = pred.index(tokenizer.eos_token_id)
        except ValueError:
            eos_pos = len(pred)-1
        pred = tokenizer.convert_ids_to_tokens(pred[bos_pos+1:eos_pos])
        y_pred.append(pred)
 
        eos_pos = target.index(tokenizer.eos_token_id)
        target = tokenizer.convert_ids_to_tokens(target[bos_pos+1:eos_pos])
        y.append(target)
        
    return y_pred, y

def flatten_logit_label(logits, label):
    lm_logits_flat_shifted = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    lm_labels_flat_shifted = label[..., :-1].contiguous().view(-1)
    return lm_logits_flat_shifted, lm_labels_flat_shifted
        

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/pretrainDial.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./data/dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model") # facebook/bart-base
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="O2", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    if "bart" in args.model_checkpoint or "facebook" in args.model_checkpoint:
        tokenizer_class = BartTokenizer
        model_class = BartForConditionalGeneration
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        args.transformer = True
    elif "gpt2" in args.model_checkpoint:
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2LMHeadModel
        args.transformer = False
    else:
        tokenizer_class = OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
        model_class = OpenAIGPTLMHeadModel
        args.transformer = False
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

    # model_class = GPT2LMHeadModel if "gpt2" in args.model_checkpoint else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint, max_length=512)
    model.to(args.device)

    if "bart" in args.model_checkpoint:
        # model.config.max_length = 256
        decoder_start_token = tokenizer.decode([model.config.decoder_start_token_id])
        SPECIAL_TOKENS = ["<usr>", "<sys>", "<s>", "</s>", "", "", "<pad>"]
        ATTR_TO_SPECIAL_TOKEN = {'bos_token': "<s>", 'eos_token': "</s>", 'pad_token': '<pad>',
                                'additional_special_tokens': ['<usr>', '<sys>']}
        
    else:
        SPECIAL_TOKENS = ["<usr>", "<sys>", "<bos>", "<eos>", "<ctx>", "</ctx>", "<pad>"]
        ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                                'additional_special_tokens': ['<usr>', '<sys>','<ctx>','</ctx>']}
    
    # Add special tokens if they are notready added
    add_special_tokens_(model, tokenizer, ATTR_TO_SPECIAL_TOKEN)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    logger.info("==========SPECIAL TOKENS==========")
    logger.info(SPECIAL_TOKENS)
    special_token_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    logger.info(special_token_ids)
    logger.info("==================================")

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer,SPECIAL_TOKENS)
    

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.long().to(args.device) for input_tensor in batch)
        if len(batch)==3:
            input_ids, token_type_ids, lm_labels = batch
            loss, *_ = model(input_ids, token_type_ids=token_type_ids, labels=lm_labels)
        else:
            input_ids, dec_ids, enc_attn_mask, dec_attn_mask, lm_labels = batch
            # print(input_ids.shape, lm_labels.shape)
            loss, *_ = model(input_ids, decoder_input_ids=dec_ids, attention_mask=enc_attn_mask,\
                             decoder_attention_mask=dec_attn_mask, labels=lm_labels)
                
        loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward(),
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.long().to(args.device) for input_tensor in batch)
            if len(batch)==3:
                input_ids, token_type_ids, lm_labels = batch
                # if we dont send labels to model, it doesnt return losses
                lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
                try:
                    bos_idx = input_ids[0].tolist().index(tokenizer.bos_token_id)
                    logger.info(tokenizer.decode(input_ids[0,:bos_idx+1].tolist()))
                    logger.info(re.sub(r'<pad>','',tokenizer.decode(torch.argmax(lm_logits[0,bos_idx:-1,:], dim=-1).tolist())))
                    logger.info("----------------------------------------------------")
                except TypeError:
                    pass
            else:
                input_ids, dec_ids, enc_attn_mask, dec_attn_mask, lm_labels = batch
                # if we dont send labels to model, it doesnt return losses
                loss, lm_logits, *_ = model(input_ids, decoder_input_ids=dec_ids, attention_mask=enc_attn_mask,\
                            decoder_attention_mask=dec_attn_mask, labels=lm_labels)
                try:
                    logger.info(re.sub(r'<pad>','',tokenizer.decode(torch.argmax(lm_logits[0,:-1,:], dim=-1).tolist())))
                    logger.info(re.sub(r'<pad>','',tokenizer.decode(dec_ids[0].tolist())))
                    logger.info("----------------------------------------------------")
                except TypeError:
                    pass

            # lm_logits = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            # lm_labels = lm_labels[..., :-1].contiguous().view(-1)
            # return lm_logits_flat_shifted, lm_labels_flat_shifted
            return lm_logits[...,:-1,:], lm_labels[..., 1:]

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=4000), lambda _:evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    
    # metrics = {"bleu_4": Bleu(smooth="smooth1", output_transform=lambda x: decode_logit_label(x[0],x[1], tokenizer))}
    metrics={"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: flatten_logit_label(x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics['nll'], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=5)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=15000), checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
