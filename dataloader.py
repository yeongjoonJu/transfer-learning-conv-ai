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

def build_input_from_segments_bart(dialog, tokenizer, special_tokens, seq_len=512, separate=False):
    """ Decorate the sequence with additional tokens. """
    """reference: https://github.com/haven-jeon/KoBART-chatbot/blob/main/kobart_chit_chat.py"""
    bos, eos, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    # Generate context
    indices_ctx = [bos]
    for turn in dialog[:-1]:
        indices_ctx = indices_ctx + turn['usr'] + turn['sys']
    if not separate:
        indices_ctx = indices_ctx + dialog[-1]['usr']
    indices_ctx += [eos]

    if len(indices_ctx) > seq_len:
        return None

    # Generate response
    if separate:
        indices_res = [bos] + dialog[-1]['usr'] + dialog[-1]['sys'] + [eos]
        encoder_input_id, encoder_attention_mask = make_input_id_mask(indices_ctx, tokenizer, seq_len)
        decoder_input_id, decoder_attention_mask = make_input_id_mask(indices_res, tokenizer, seq_len)
        labels = [-100]*(len(dialog[-1]['usr'])+1)
        labels += dialog[-1]['sys'] +[eos]
        labels += [-100]*(seq_len-len(labels))
    else:
        indices_res = [bos] + dialog[-1]['sys'] + [eos]
        encoder_input_id, encoder_attention_mask = make_input_id_mask(indices_ctx, tokenizer, seq_len)
        decoder_input_id, decoder_attention_mask = make_input_id_mask(indices_res, tokenizer, seq_len//2)
        labels = indices_res.copy()
        labels += [-100]*(seq_len//2-len(labels))

    return {'input': encoder_input_id, 'attention_mask': encoder_attention_mask,\
            'decoder_input_ids': decoder_input_id, 'decoder_attention_mask':decoder_attention_mask, 'labels':labels[1:]+[-100]}


def build_input_from_segments(dialog, tokenizer, special_tokens, seq_len=512):
    """ Decorate the sequence with additional tokens for GPT. """
    eos, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])
    induction_token_id = tokenizer.convert_tokens_to_ids(['=>'])
    
    # Generate context
    indices_ctx = []
    ctx_token_types = []
    for turn in dialog[:-1]:
        indices_ctx = indices_ctx + turn['usr'] + turn['sys']
        ctx_token_types = ctx_token_types + user_token_id*len(turn['usr']) + system_token_id*len(turn['sys'])
    indices_ctx = indices_ctx + dialog[-1]['usr']
    ctx_token_types = ctx_token_types + user_token_id*len(dialog[-1]['usr'])

    # Generate response
    indices_res = dialog[-1]['sys'] + [eos]

    # context => response
    source = indices_ctx + induction_token_id + indices_res
    token_id = ctx_token_types + system_token_id*len(induction_token_id) + len(indices_res) * system_token_id
    target = [-100]*len(indices_ctx) + [-100]*len(induction_token_id) + indices_res

    if len(source) < seq_len:
        source, attention_mask = make_input_id_mask(source, tokenizer, seq_len)
        target += [-100]*(seq_len-len(target))
        token_id += [0]*(seq_len-len(token_id))
    else:
        attention_mask = [1] * seq_len
        source = source[-seq_len:]
        target = target[-seq_len:]
        token_id = token_id[-seq_len:]

    return {'input': source, 'token_type_ids': token_id, 'mask':attention_mask, 'output': target}
    

def tokenize_multi_turn_dialog(dataset, tokenizer, special_tokens):
    """
    Format > [[{'usr': <user utterance>, 'sys': <system utterance>}, ...],...]
    """
    tokenized_dialogs = []
    for i, dialog in enumerate(dataset):
        print('\r %.4f...' % ((i+1)/len(dataset)), end='')
        tokenized_turn = []
        for turn in dialog:
            usr_ut = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('user : ' + turn['usr']))
            sys_ut = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('system : ' + turn['sys']))
            tokenized_turn.append({'usr': usr_ut, 'sys': sys_ut})
            
            # Save all dialogues per turn
            tokenized_dialogs.append(tokenized_turn.copy())
    print('\nDone!')

    return tokenized_dialogs


def get_dataset(tokenizer, dataset_path, special_tokens, split_rate, seq_len, dataset_cache='./data/predial_cache', separate=False, model_name=None):
    """ Get dataset from json or cache."""

    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    sep = "_sep" if separate else ""
    
    if dataset_cache and os.path.isfile(dataset_cache+sep+'_train') and os.path.isfile(dataset_cache+sep+'_val'):
        print("Load tokenized dataset from cache at %s", dataset_cache + sep)
        train_dataset = torch.load(dataset_cache + sep + '_train')
        val_dataset = torch.load(dataset_cache + sep + '_val')
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
        print("Model:", model_name)
        dataset = train_dataset+val_dataset
        print(len(train_dataset),'+',len(val_dataset),'=',len(train_dataset)+len(val_dataset))

        input_tensors = []
        output_tensors = []
        if "gpt2" in model_name:
            token_type_tensors = []
            mask_tensors = []
            for dialog in tqdm(dataset):
                dialog = build_input_from_segments(dialog, tokenizer, special_tokens, seq_len=seq_len)
                input_tensors.append(torch.LongTensor(dialog['input']).unsqueeze(0))
                output_tensors.append(torch.LongTensor(dialog['output']).unsqueeze(0))
                mask_tensors.append(torch.LongTensor(dialog['mask']).unsqueeze(0))
                token_type_tensors.append(torch.LongTensor(dialog['token_type_ids']).unsqueeze(0))

            train_dataset = TensorDataset(torch.cat(input_tensors[:len_train]), torch.cat(token_type_tensors[:len_train]), torch.cat(mask_tensors[:len_train]), torch.cat(output_tensors[:len_train]))
            val_dataset = TensorDataset(torch.cat(input_tensors[len_train:]), torch.cat(token_type_tensors[len_train:]), torch.cat(mask_tensors[len_train:]), torch.cat(output_tensors[len_train:]))
        else:
            enc_attn_tensors = []
            dec_attn_tensors = []
            dec_input_tensors = []
            for dialog in tqdm(dataset):
                dialog = build_input_from_segments_bart(dialog, tokenizer, special_tokens, separate=separate, seq_len=seq_len)
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

        dataset_cache = dataset_cache + sep
        torch.save(train_dataset, dataset_cache+'_train')
        torch.save(val_dataset, dataset_cache+'_val')

    return train_dataset, val_dataset


def get_data_loaders(args, tokenizer, special_tokens, split_rate=0.05):
    """ Prepare the dataset for training and evaluation """
    train_dataset, valid_dataset = get_dataset(tokenizer, args.dataset_path, special_tokens, split_rate, args.max_seq_len, \
                                                 args.dataset_cache, separate=args.separate, model_name=args.model_checkpoint,)

    print("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, num_workers=1, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, num_workers=1, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    print("Train dataset (Samples, Seq length): {}".format(train_dataset.tensors[0].shape))
    print("Valid dataset (Samples, Seq length): {}".format(valid_dataset.tensors[0].shape))
    
    return train_loader, valid_loader, train_sampler, valid_sampler
