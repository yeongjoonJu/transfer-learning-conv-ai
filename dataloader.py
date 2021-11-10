# from datetime import datetime
import json
from numpy.core.fromnumeric import sum
from numpy.core.numeric import indices
# import logging
import os
# import tarfile
# import tempfile
# import socket
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Queue

import torch
# from torch._C import LongTensor
from torch.functional import Tensor
from transformers import cached_path
from torch.utils.data import TensorDataset, DataLoader

from transformers import (AdamW, BartForConditionalGeneration, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, BartTokenizer, BartModel,
                                  GPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

def make_input_id_mask(input_id, tokenizer, max_seq_len=512):
    attention_mask = [1] * len(input_id) + [0]*(max_seq_len-len(input_id))
    input_id += [tokenizer.pad_token_id]*(max_seq_len-len(input_id))

    return input_id, attention_mask

def build_input_from_segments_bart(dialog, tokenizer, special_tokens, seq_len=512):
    """ Decorate the sequence with additional tokens. """
    usr, sys, bos, eos, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    # Generate context
    indices_ctx = [bos]
    for turn in dialog[:-1]:
        indices_ctx = indices_ctx + turn['usr'] + turn['sys']
    indices_ctx = indices_ctx + dialog[-1]['usr'] + [eos]

    if len(indices_ctx)  > seq_len:
        return None

    # Encoder input : <s> <usr> Hello, How are you? <sys> yes, sure <usr> Thanks </s>
    # Decoder input : <s> Good! </s>

    # Generate response
    indices_res = [bos] + dialog[-1]['sys'][1:] + [eos]

    encoder_input_id, encoder_attention_mask = make_input_id_mask(indices_ctx, tokenizer, seq_len)
    decoder_input_id, decoder_attention_mask = make_input_id_mask(indices_res, tokenizer, seq_len//2)
    labels = indices_res[1:(seq_len//2+1)]
    labels += [-100]*(seq_len//2-len(labels))
    return {'input': encoder_input_id, 'attention_mask': encoder_attention_mask,\
            'decoder_input_ids': decoder_input_id, 'decoder_attention_mask':decoder_attention_mask, 'labels':labels}


def build_input_from_segments(dialog, tokenizer, special_tokens, separate_request=False, transformer=False, seq_len=512):
    """ Decorate the sequence with additional tokens. """
    usr, sys, res, eos, b_ctx, e_ctx, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    # <ctx> <usr> ~~~ <sys> ~~~~ <usr> ~~~~ </ctx> <bos> ~~~ <eos>
    
    # pad for remaining sequence   
    if transformer:
        # Generate context
        indices_ctx = []
        for turn in dialog[:-1]:
            indices_ctx = indices_ctx + turn['usr'] + turn['sys']
        indices_ctx = indices_ctx + dialog[-1]['usr']
        
        # Generate response
        indices_res = [res] + dialog[-1]['sys'][1:] + [eos]

        if len(indices_ctx)  > seq_len:
            return None
        ctx_pad_len = seq_len - len(indices_ctx)
        attention_mask = [1]*len(indices_ctx) + [0]*ctx_pad_len
        indices_ctx += [pad] * ctx_pad_len
        indices_res_label = indices_res + [-100] * (seq_len//2 - len(indices_res))
        indices_res += [pad] * (seq_len//2 - len(indices_res))
        return {'input': indices_ctx, 'attention_mask':attention_mask, 'dec_input': indices_res[:-1], 'output': indices_res_label[1:]}
    else:
        # Generate context
        indices_ctx = [b_ctx]
        token_type_ids = [0]
        for turn in dialog[:-1]:
            indices_ctx = indices_ctx + turn['usr'] + turn['sys']
            token_type_ids = token_type_ids + [1]*len(turn['usr']) + [0]*len(turn['sys'])
        indices_ctx = indices_ctx + dialog[-1]['usr'] + [e_ctx]
        token_type_ids = token_type_ids + [1]*len(dialog[-1]['usr']) + [0]

        # Generate response
        indices_res = [res] + dialog[-1]['sys'][1:] + [eos]
        token_type_ids = token_type_ids + [0]*len(indices_res)

        if (len(indices_ctx) + len(indices_res)) > seq_len:
            return None
            
        # Prevent pad from backpropagation
        ctx_sep = [-100] * len(indices_ctx)
        pad_len = seq_len - (len(indices_ctx)+len(indices_res))
        token_type_ids = token_type_ids + [0]*pad_len
        indices_res_label = indices_res + [-100] * pad_len
        indices_res += [pad] * pad_len
                
        return {'input': indices_ctx + indices_res[:-1], 'token_type_ids': token_type_ids[:-1], 'output': ctx_sep[1:]+indices_res_label}
    

def tokenize_multi_turn_dialog(dataset, tokenizer, special_tokens):
    """
    Format > [[{'usr': <user utterance>, 'sys': <system utterance>}, ...],...]
    """
    usr, sys, res, eos, _, _, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    tokenized_dialogs = []
    for i, dialog in enumerate(dataset):
        print('\r %.4f...' % ((i+1)/len(dataset)), end='')
        tokenized_turn = []
        for turn in dialog:
            usr_ut = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn['usr']))
            sys_ut = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn['sys']))
            tokenized_turn.append({'usr': [usr] + usr_ut, 'sys': [sys] + sys_ut})
            
            # Save all dialogues per turn
            tokenized_dialogs.append(tokenized_turn.copy())
    print('\nDone!')

    return tokenized_dialogs


def get_dataset(tokenizer, dataset_path, special_tokens, split_rate, dataset_cache='./data/predial_cache'):
    """ Get dataset from json or cache."""
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache+'_train') and os.path.isfile(dataset_cache+'_val'):
        print("Load tokenized dataset from cache at %s", dataset_cache)
        train_dataset = torch.load(dataset_cache + '_train')
        val_dataset = torch.load(dataset_cache + '_val')
    else:
        if os.path.exists(dataset_cache + '_train_ids.json'):
            with open(dataset_cache + '_train_ids.json', 'r', encoding='utf-8') as f:
                train_dataset = json.load(f)
            with open(dataset_cache + '_val_ids.json', 'r', encoding='utf-8') as f:
                val_dataset = json.load(f)
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            
            print('The total dataset:', len(dataset))

            print("Tokenize and encode the dataset")
            np.random.shuffle(dataset)
            len_val = int(len(dataset) * split_rate)
            val_dataset = dataset[:len_val]
            train_dataset = dataset[len_val:]

            # Tokenize
            train_dataset = tokenize_multi_turn_dialog(train_dataset, tokenizer, special_tokens)
            val_dataset = tokenize_multi_turn_dialog(val_dataset, tokenizer, special_tokens)

            with open(dataset_cache + '_train_ids.json', 'w') as f:
                json.dump(train_dataset, f, ensure_ascii=True)
            with open(dataset_cache + '_val_ids.json', 'w') as f:
                json.dump(val_dataset, f, ensure_ascii=True)

        len_train = len(train_dataset)

        # Caching
        print('Tokenized dataset:', len(train_dataset) + len(val_dataset))
        print("Pad inputs and convert to Tensor")
        dataset = train_dataset+val_dataset

        input_tensors = []
        output_tensors = []
        if type(tokenizer).__name__[:4]=='GPT2':    
            token_type_tensors = []
            for dialog in tqdm(dataset):
                dialog = build_input_from_segments(dialog, tokenizer, special_tokens, transformer=False)
                if dialog is None:
                    continue
                input_tensors.append(torch.LongTensor(dialog['input']).unsqueeze(0))
                output_tensors.append(torch.LongTensor(dialog['output']).unsqueeze(0))
                token_type_tensors.append(torch.LongTensor(dialog['token_type_ids']).unsqueeze(0))

            train_dataset = TensorDataset(torch.cat(input_tensors[:len_train]), torch.cat(token_type_tensors[:len_train]), torch.cat(output_tensors[:len_train]))
            val_dataset = TensorDataset(torch.cat(input_tensors[len_train:]), torch.cat(token_type_tensors[len_train:]), torch.cat(output_tensors[len_train:]))
        else:
            enc_attn_tensors = []
            dec_attn_tensors = []
            dec_input_tensors = []
            for dialog in tqdm(dataset):
                dialog = build_input_from_segments_bart(dialog, tokenizer, special_tokens)
                if dialog is None:
                    continue
                input_tensors.append(torch.LongTensor(dialog['input']).unsqueeze(0))
                enc_attn_tensors.append(torch.LongTensor(dialog['attention_mask']).unsqueeze(0))
                dec_attn_tensors.append(torch.LongTensor(dialog['decoder_attention_mask']).unsqueeze(0))
                dec_input_tensors.append(torch.LongTensor(dialog['decoder_input_ids']).unsqueeze(0))
                output_tensors.append(torch.LongTensor(dialog['labels']).unsqueeze(0))
                
            train_dataset = TensorDataset(torch.cat(input_tensors[:len_train]), torch.cat(dec_input_tensors[:len_train]),\
                                          torch.cat(enc_attn_tensors[:len_train]), torch.cat(dec_attn_tensors[:len_train]),\
                                          torch.cat(output_tensors[:len_train]))
            val_dataset = TensorDataset(torch.cat(input_tensors[len_train:]), torch.cat(dec_input_tensors[len_train:]),\
                                          torch.cat(enc_attn_tensors[len_train:]), torch.cat(dec_attn_tensors[len_train:]),\
                                          torch.cat(output_tensors[len_train:]))

        print('\n======> Done!')

        torch.save(train_dataset, dataset_cache+'_train')
        torch.save(val_dataset, dataset_cache+'_val')

    return train_dataset, val_dataset


def get_data_loaders(args, tokenizer, special_tokens, split_rate=0.05):
    """ Prepare the dataset for training and evaluation """
    train_dataset, valid_dataset = get_dataset(tokenizer, args.dataset_path, special_tokens, split_rate, args.dataset_cache)

    print("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, num_workers=1, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, num_workers=1, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    print("Train dataset (Samples, Seq length): {}".format(train_dataset.tensors[0].shape))
    print("Valid dataset (Samples, Seq length): {}".format(valid_dataset.tensors[0].shape))
    
    return train_loader, valid_loader, train_sampler, valid_sampler