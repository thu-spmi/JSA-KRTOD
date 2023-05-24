"""
Copyright 2022 Tsinghua University
Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
"""

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertForPreTraining, BertForNextSentencePrediction
from transformers import BertTokenizer
from reader import *
from metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, CrossEntropyLoss, CosineEmbeddingLoss, CosineSimilarity
from torch.utils.data import DataLoader

import os, shutil
import random
import argparse
import time
import logging
import json
from tqdm import tqdm
import numpy as np
import copy, re
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg

class EBM(torch.nn.Module):
    def __init__(self, cfg, tokenizer):
        super(EBM, self).__init__()
        self.cfg = cfg
        self.bert_model=BertModel.from_pretrained('bert-base-chinese')#cfg.model_path
        self.bert_model.resize_token_embeddings(len(tokenizer))
        self.dropout = Dropout(cfg.dropout)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, 1) # we need to get an energy function, returning the logit
        if cfg.add_extra_feature:
            self.reg_weight = nn.Linear(1, 1) # cfg.reg_weight

    def forward(self,input_ids: torch.tensor,
                attention_mask: torch.tensor, feature=None):
        hidden_states = self.bert_model(input_ids=input_ids,attention_mask = attention_mask)[0]
        pooled_output =  hidden_states[:, :].mean(dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        if feature is not None:
            logits = logits + self.reg_weight(feature.unsqueeze(-1))
            #for i in range(len(logits)):
            #    logits[i][0] = logits[i][0] + self.reg_weight(feature[i]) 
        #logits = self.classifier(pooled_output)
        return logits

class Bert_Model(torch.nn.Module): # not used, do not copy
    def __init__(self, cfg, tokenizer):
        super(Bert_Model, self).__init__()
        self.cfg = cfg
        self.bert_model=BertForPreTraining.from_pretrained('bert-base-chinese')#cfg.model_path
        self.bert_model.resize_token_embeddings(len(tokenizer))
        #self.dropout = Dropout(cfg.dropout)
        #self.classifier = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size) 
        #self.num_labels = cfg.num_domains

    def forward(self,input_ids: torch.tensor,
                attention_mask: torch.tensor,
                label: torch.tensor = None):
        hidden_states = self.bert_model(input_ids=input_ids,attention_mask = attention_mask)[0]
        pooled_output =  hidden_states[:, :].mean(dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), label.type(torch.long))
        return logits, loss

class EncoderModel(torch.nn.Module):
    def __init__(self, cfg, tokenizer):
        super(EncoderModel, self).__init__()
        self.sentence_model=BertModel.from_pretrained('bert-base-chinese')
        self.triple_model=BertModel.from_pretrained('bert-base-chinese')
        self.inner_product = nn.Bilinear(self.sentence_model.config.hidden_size, self.triple_model.config.hidden_size, 1) # 768
        self.sentence_model.resize_token_embeddings(len(tokenizer))
        self.triple_model.resize_token_embeddings(len(tokenizer))
        if isinstance(cfg.device,list):
            self.device = cfg.device[0]
            self.device1 = cfg.device[-1]
        else:
            self.device = cfg.device
            self.device1 = cfg.device
        
        self.sentence_model.to(self.device)
        self.inner_product.to(self.device)
        self.triple_model.to(self.device1)

        if 'train' in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)        
        # log
        log_path='./log/log_retrieve/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)

    def forward(self,input_sent: torch.tensor,
                attention_sent: torch.tensor,
                input_triple,
                attention_triple,
                label):
        THRESHOLD = 0.5
        hidden_states = self.sentence_model(input_ids=input_sent,attention_mask = attention_sent)[0]
        h_sent =  hidden_states[:, :].mean(dim=1) # [cls] is also ok
        hidden = self.sentence_model(input_ids=input_triple,attention_mask = attention_triple)[0] # to triple's device
        h_triple =  hidden[:, :].mean(dim=1) # [cls] is also ok
        # cos_sim = CosineSimilarity(dim=1)
        # sim = cos_sim(h_sent, h_triple) # sim for score
        # cos_loss = CosineEmbeddingLoss() # margin = cfg.margin
        # loss = cos_loss(h_sent, h_triple, label) # can try Bilinear discriminator and celoss
        logits = self.inner_product(h_sent, h_triple) # need to be on the same device
        # loss_fct = nn.CrossEntropyLoss(reduction='sum') # 
        loss_fct = nn.BCELoss(reduction='sum')
        probs = torch.sigmoid(logits)
        # loss = nn.BCEWithLogitsLoss(logits, label.unsqueeze(-1).float())
        loss = loss_fct(probs, label.unsqueeze(-1).float())
        predictions = (probs<(1-THRESHOLD)).squeeze()
        labels = (label==0)
        accuracy = (predictions == labels).sum() / label.shape[0]
        
        return loss, accuracy.item(), predictions.cpu().tolist(), labels.cpu().tolist()

def get_optimizers(num_samples, model, lr): # , cfg
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    #print(num_samples, cfg.epoch_num, cfg.gradient_accumulation_steps, cfg.batch_size)
    num_training_steps = num_samples*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size)
    num_warmup_steps = int(num_training_steps*cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
        num_training_steps=num_training_steps)
    return optimizer, scheduler

def collate_fn_ebm(batch):
    collated = {}
    if batch!= []:
        for k,_ in batch[0].items():
            collated[k] = [b[k] for b in batch]
    return collated

def collate_fn(batch):
    pad_id = cfg.pad_id
    pad_result = {}
    label_shift = {1:0, -1:1}
    # label_shift2 = {1:1, -1:0}
    if cfg.only_one_model:
        wanted_key = ['input']
        for s  in batch:
            s['input'] = s['context'] + s['triple'][1:]
            sep_id = s['context'][-1]
    else:
        wanted_key = ['context', 'triple']
    if batch!= []:
        for key in wanted_key:# add padding for input, ouput and attentions
            #np.array(
            #attention=len(encoded)*[1]
            #if  not isinstance(self[0][key],int): 
            max_len = max(len(input[key]) for input in batch)
            max_len = min(max_len, cfg.max_sequence_len)
            pad_batch=np.ones((len(batch), max_len))*pad_id  #-100
            pad_attention_batch=np.ones((len(batch), max_len))*pad_id
            pad_type_batch=np.ones((len(batch), max_len))
            for idx, s in enumerate(batch):
                #trunc = s[-max_len:]
                if len(s[key])>cfg.max_sequence_len:
                    pad_batch[idx, :max_len] = np.array(s[key][-max_len:])
                    pad_attention_batch[idx, :max_len] = np.ones(max_len)
                else:
                    pad_batch[idx, :len(s[key])] = np.array(s[key])
                    pad_attention_batch[idx, :len(s[key])] = np.ones(len(s[key]))
                if cfg.only_one_model:
                    pad_type_batch[idx, :s[key].index(sep_id)] = np.ones(s[key].index(sep_id) if s[key].index(sep_id)<max_len else max_len)*0 # need more care afterwards
            pad_result[(key)] = torch.from_numpy(pad_batch).long()
            pad_result[(key+'_attention')] = torch.from_numpy(pad_attention_batch).long()
        if cfg.only_one_model:
            pad_result[(key+'_type')] = torch.from_numpy(pad_type_batch).long()
        
        if 'label' in batch[0]:
            pad_batch=np.ones(len(batch))
            for idx, s in enumerate(batch):
                #if cfg.only_one_model:
                pad_batch[idx] = label_shift[s['label']] # if cfg.only_one_model else s['label']
                #else:
                #   pad_batch[idx] = label_shift2[s['label']]
            pad_result['label'] = torch.from_numpy(pad_batch).long()
    return pad_result

def train(cfg):
    cfg.exp_path = 'experiments_retrieve'
    cfg.batch_size = 16 # 32
    cfg.lr = 1e-5

    json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)   
    # Add special tokens
    init_vocab_size=len(tokenizer)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    logging.info('Added special tokens:{}'.format(special_tokens))
    tokenizer.add_special_tokens(special_tokens_dict)
    logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(tokenizer)))

    encoded_data = read_data(tokenizer, retrieve=True)
    if cfg.only_one_model:
        model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')#EncoderModel(cfg,tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        model.to(cfg.device[0])
    else:
        model = EncoderModel(cfg,tokenizer)

    train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn) 
    dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
   
    optimizer, scheduler = get_optimizers(num_samples=len(encoded_data['train']) ,model=model, lr=cfg.lr)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    for epoch in range(cfg.epoch_num):
        model.train()
        epoch_loss = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(cfg.device[0])
                if cfg.only_one_model:
                    loss = model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"],labels=batch["label"]).loss
                else:
                    loss = model(input_sent=batch["context"], attention_sent=batch["context_attention"], input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])[0]
                loss.backward()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        print(epoch_loss / num_batches)
        # Evaluate and save checkpoint
        score, precision, recall, f1 = evaluate(model, test_dataloader, cfg) # dev_dataloader
        metrics_to_log["eval_score"] = score
        logging.info("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        s = score + recall + f1
        print("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        if s > best_score:
            logging.info("New best results found! Score: {}".format(score))
            #model.bert_model.save_pretrained(cfg.save_dir)
            if cfg.only_one_model:
                if not os.path.exists(cfg.bert_save_path):
                    os.mkdir(cfg.bert_save_path)
                model.save_pretrained(cfg.bert_save_path)
                tokenizer.save_pretrained(cfg.bert_save_path)
            else:
                if not os.path.exists(cfg.context_save_path):
                    os.mkdir(cfg.context_save_path)
                if not os.path.exists(cfg.triple_save_path):
                    os.mkdir(cfg.triple_save_path)
                if not os.path.exists(cfg.retrieval_save_path):
                    os.mkdir(cfg.retrieval_save_path)
                model.sentence_model.save_pretrained(cfg.context_save_path)
                model.triple_model.save_pretrained(cfg.triple_save_path)
                tokenizer.save_pretrained(cfg.context_save_path)
                tokenizer.save_pretrained(cfg.triple_save_path)
                torch.save(model.state_dict(), os.path.join(cfg.retrieval_save_path, "model.pt"))
            best_score = s
    #model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "model.pt")))
    #score = evaluate(model, test_dataloader, cfg)
    #print(score)
    return

def train_ebm(cfg):
    cfg.exp_path = 'experiments_retrieve_ebm'
    cfg.lr = 5e-6
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)
    json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)   
    # Add special tokens
    init_vocab_size=len(tokenizer)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    logging.info('Added special tokens:{}'.format(special_tokens))
    # do
    tokenizer.add_special_tokens(special_tokens_dict)
    logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(tokenizer)))
    model = EBM(cfg,tokenizer)
    model.to(cfg.device[0])
    if cfg.only_one_model:
        proposal_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
        proposal_model.to(cfg.device[-1])
    encoded_data = read_data(tokenizer, ebm=True)
    if cfg.debugging:
        train_dataloader=DataLoader(encoded_data['train'][:400], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_ebm) 
    else:
        train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_ebm)
    dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
   
    optimizer, scheduler = get_optimizers(num_samples=len(encoded_data['train']) ,model=model, lr=cfg.lr)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    nan_num = 0
    for epoch in range(cfg.epoch_num):
        statistic = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        statistic_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        model.train()
        epoch_loss = 0
        gt_energy = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # loss from data
            triples_gt = []

            try:  # avoid OOM
                positive_count = []
                for cases in batch['cases']:
                    gt_seq = ''
                    tmp_count = 0
                    for c in cases:
                        if c['label']==1:
                            gt_seq = gt_seq + tokenizer.decode(c['triple']).replace('[CLS]','').replace('[SEP]','') + '；'
                            tmp_count += 1 
                    if '；' in gt_seq:
                        gt_seq = gt_seq[:-1] 
                    triples_gt.append(gt_seq)
                    positive_count.append(tmp_count)
                gt_input = get_retrieval_sequence(tokenizer, batch['context'], triples_gt)
                gt_input.to(cfg.device[0])
                positive_counts = torch.tensor(positive_count, dtype=torch.float).to(cfg.device[0])
                if cfg.add_extra_feature:
                    logits = model(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"], feature=positive_counts)
                else:
                    logits = model(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    loss = 0.0
                    print(logits)
                    nan_num = nan_num + 1
                else:
                    loss = -sum(logits)
                    gt_energy += loss.item()
                #if loss.item()>-2000:
                #    print('warning, energy to large')
                # loss = sum(math.exp(-logits)), incorrect as the loss is the -logp
                # loss from MIS
                # get_probability
                    statistic[0] += loss.item()
                    statistic_count[0] +=cfg.batch_size # batch_size
                for i in range(len(batch['cases'])):
                    p_batch = collate_fn(batch['cases'][i])
                    for key, val in p_batch.items():
                        if type(p_batch[key]) is list:
                            continue
                        p_batch[key] = p_batch[key].to(cfg.device[-1])
                    if p_batch!={}:
                        logits = proposal_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                        probs = F.softmax(logits, dim=1)
                        accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                        gt_label = p_batch['label'].cpu().tolist() 
                    else:
                        accept_prob = []
                    triple_num = len(accept_prob)
                    # sampling
                    if cfg.train_ebm_mis and 'cached' in batch['cases'][i][0]:
                        proposals = [batch['cases'][i][0]['cached'][0]]
                        proposal_log_probs = [batch['cases'][i][0]['cached'][1]]
                        proposal_wrong_num = [batch['cases'][i][0]['cached'][2]]
                    else:
                        proposals = []
                        proposal_log_probs = []
                        proposal_wrong_num = []
                    for sample_num in range(cfg.train_sample_num):
                        p_prob = 0.0
                        proposal = []
                        proposal_id = []
                        random.random()
                        for num in range(triple_num):
                            p = random.random()
                            if p< accept_prob[num]:
                                proposal.append(tokenizer.decode(batch['cases'][i][num]['triple'][1:-1]).replace(' ','')) # can be unified to the .replace('[CLS]','').replace('[SEP]','')
                                # can directly concatenate all the triples to improve efficiency
                                p_prob += math.log(accept_prob[num])
                                proposal_id.append(0)
                            else:
                                p_prob += math.log(1-accept_prob[num])
                                proposal_id.append(1)
                        if proposal_id!=gt_label or cfg.use_all_proposal: #use cfg.use_all_proposal to include gt_label to be trained
                            proposals.append('；'.join(proposal))
                            proposal_log_probs.append(p_prob)
                            proposal_wrong_num.append(sum(gt_label[i]!=proposal_id[i] for i in range(len(gt_label))))
                    # get IS_loss, avoiding OOM
                    is_logits = []
                    sample_num = len(proposals)
                    positive_count = torch.tensor([float(len(item.split('；'))) for item in proposals], 
                    dtype=torch.float).to(cfg.device[0])
                    if sample_num>0:
                        for b_num in range(cfg.train_sample_times): # cfg.train_sample_times>1 need to be modified
                            input = get_retrieval_sequence(tokenizer, [batch['context'][i]]*sample_num, proposals[b_num*16 : (b_num+1)*16])
                            input.to(cfg.device[0])
                            if cfg.add_extra_feature:
                                is_logits.extend(model(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count))
                            else:
                                is_logits.extend(model(input_ids=input['input_ids'], attention_mask=input["attention_mask"]))
                        is_ratio = []
                        for j in range(sample_num):
                            if (-proposal_log_probs[j] + is_logits[j].item())>200:
                                is_ratio.append(math.exp(200))
                                print('large is ratio found')
                                nan_num = nan_num + 1
                            else:
                                if cfg.residual:
                                    is_ratio.append(math.exp (is_logits[j].item()))
                                else:
                                    is_ratio.append(math.exp(-proposal_log_probs[j] + is_logits[j].item()))
                        if cfg.train_ebm_mis:
                            mis_results = {}
                            max = is_ratio[0]
                            current = 0
                            lengths = 0
                            #mis_results[0] = 0
                            for j in range(sample_num):
                                tmp_prob = random.random()
                                if tmp_prob<(is_ratio[j]/max): # is_ratio[j]> max,
                                    # actually max is not necessarily the current max
                                    mis_results[current] = lengths
                                    max = is_ratio[j]
                                    current = j
                                    lengths = 1
                                else:
                                    lengths += 1
                            #if current==0:
                            #    mis_results[0] = lengths
                            mis_results[current] = lengths
                            # sample should be added
                            normalize = sum(mis_results[tmp] for tmp in mis_results) 
                            # mis performs averaging instead of weighted averaging
                            batch['cases'][i][0]['cached']=(proposals[j], proposal_log_probs[j], proposal_wrong_num[j])
                            # save cached results
                        else:
                            normalize = sum(is_ratio)
                        if normalize>0.0:
                            if cfg.train_ebm_mis:    
                                for index, length in mis_results.items():
                                    if proposal_wrong_num[index] in statistic:
                                        statistic[proposal_wrong_num[index]] += is_logits[index].item()
                                        statistic_count[proposal_wrong_num[index]] += 1
                                    else:
                                        statistic[5] += is_logits[index].item()
                                        statistic_count[5] += 1
                                    loss = loss + (length*is_logits[index])/normalize
                            else:
                                for j in range(sample_num):
                                    if proposal_wrong_num[j] in statistic:
                                        statistic[proposal_wrong_num[j]] += is_logits[j].item()
                                        statistic_count[proposal_wrong_num[j]] += 1
                                    else:
                                        statistic[5] += is_logits[j].item()
                                        statistic_count[5] += 1
                                    if cfg.reject_control:
                                        if is_ratio[j]/normalize>0.003: # add reject control
                                            loss = loss + (is_ratio[j]* is_logits[j])/normalize
                                        else:
                                            if random.random()<is_ratio[j]*200/normalize:
                                                loss = loss + 0.005*is_logits[j]
                                    else:
                                        loss = loss + is_ratio[j]*is_logits[j]/normalize
                if loss!=0.0 and (not torch.isnan(loss).any()) and (not torch.isinf(loss).any()):
                    loss.backward()
                    epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory, batch size: {}".format(cfg.batch_size))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        tmp = [statistic[i]/statistic_count[i] for i in statistic]
        print(tmp)
        print(epoch_loss / num_batches)
        print(gt_energy / num_batches)
        # Evaluate and save checkpoint
        if not cfg.debugging:
            score, precision, recall, f1 = rescore(proposal_model, model, test_dataloader, cfg, tokenizer) # dev_dataloader
            metrics_to_log["eval_score"] = score
            logging.info("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
            s = score + f1 # recall +
            print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
            #if (epoch-1)%5==0:
            #    score1, precision1, recall1, f11 = rescore(proposal_model, model, train_dataloader, cfg, tokenizer)
            #    print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score1, precision1, recall1, f11))
            save_path = os.path.join(cfg.ebm_save_path, f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "") + ".pt")
            print(save_path)
            if s > best_score:
                print(f"best checkpoints saved, score:{s}")
                #logging.info("New best results found! Score: {}".format(best_score))
                if not os.path.exists(cfg.ebm_save_path):
                    os.mkdir(cfg.ebm_save_path)
                tokenizer.save_pretrained(cfg.ebm_save_path)
                torch.save(model.state_dict(), save_path)
                best_score = s
        if nan_num>100:
            break
    #model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "model.pt")))
    #score = evaluate(model, test_dataloader, cfg)
    #print(score)
    return

def train_ebm_with_mis(cfg):
    cfg.exp_path = 'experiments_retrieve_ebm'
    cfg.lr = 1e-5
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)
    json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)   
    # Add special tokens
    init_vocab_size=len(tokenizer)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    logging.info('Added special tokens:{}'.format(special_tokens))
    # do
    tokenizer.add_special_tokens(special_tokens_dict)
    logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(tokenizer)))
    model = EBM(cfg,tokenizer)
    model.to(cfg.device[0])
    if cfg.only_one_model:
        proposal_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
        proposal_model.to(cfg.device[-1])
    encoded_data = read_data(tokenizer, ebm=True)

    train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_ebm) 
    dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
   
    optimizer, scheduler = get_optimizers(num_samples=len(encoded_data['train']) ,model=model, lr=cfg.lr)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    nan_num = 0
    for epoch in range(cfg.epoch_num):
        statistic = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        statistic_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        model.train()
        epoch_loss = 0
        gt_energy = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # loss from data
            triples_gt = []

            try:  # avoid OOM
                for cases in batch['cases']:
                    gt_seq = ''
                    for c in cases:
                        if c['label']==1:
                            gt_seq = gt_seq + tokenizer.decode(c['triple']).replace('[CLS]','').replace('[SEP]','') + '；'
                    if '；' in gt_seq:
                        gt_seq = gt_seq[:-1] 
                    triples_gt.append(gt_seq)
                gt_input = get_retrieval_sequence(tokenizer, batch['context'], triples_gt)
                gt_input.to(cfg.device[0])
                logits = model(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
                # get nan
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    loss = 0.0
                    print(logits)
                    nan_num += 1
                else:
                    loss = sum(logits)
                    gt_energy += loss.item()
                    statistic[0] += loss.item()
                    statistic_count[0] +=6
                # loss = sum(math.exp(-logits)), incorrect as the loss is the -logp
                # loss from MIS
                # get_probability
                for i in range(len(batch['cases'])):
                    p_batch = collate_fn(batch['cases'][i])
                    for key, val in p_batch.items():
                        if type(p_batch[key]) is list:
                            continue
                        p_batch[key] = p_batch[key].to(cfg.device[-1])
                    if p_batch!={}:
                        logits = proposal_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                        probs = F.softmax(logits, dim=1)
                        accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                        gt_label = p_batch['label'].cpu().tolist() 
                    else:
                        accept_prob = []
                    triple_num = len(accept_prob)
                    # sampling
                    proposals = []
                    proposal_log_probs = []
                    proposal_wrong_num = []
                    for sample_num in range(cfg.train_sample_num):
                        p_prob = 0.0
                        proposal = []
                        proposal_id = []
                        random.random()
                        for num in range(triple_num):
                            p = random.random()
                            if p< accept_prob[num]:
                                proposal.append(tokenizer.decode(batch['cases'][i][num]['triple'][1:-1]).replace(' ',''))
                                # can directly concatenate all the triples to improve efficiency
                                p_prob += math.log(accept_prob[num])
                                proposal_id.append(0)
                            else:
                                p_prob += math.log(1-accept_prob[num])
                                proposal_id.append(1)
                        #if proposal_id!=gt_label:
                        proposals.append('；'.join(proposal))
                        proposal_log_probs.append(p_prob)
                        proposal_wrong_num.append(sum(gt_label[i]!=proposal_id[i] for i in range(len(gt_label))))
                    # get MIS_loss, avoiding OOM
                    mis_logits = []
                    sample_num = len(proposals)
                    if sample_num>0:
                        for b_num in range(cfg.train_sample_times): # cfg.train_sample_times>1 need to be modified
                            input = get_retrieval_sequence(tokenizer, [batch['context'][i]]*sample_num, proposals[b_num*16 : (b_num+1)*16])
                            input.to(cfg.device[0])
                            mis_logits.extend(model(input_ids=input['input_ids'], attention_mask=input["attention_mask"]))
                        is_ratios = []
                        mis_results = {}
                        for j in range(sample_num):
                            if (-proposal_log_probs[j] - mis_logits[j].item())>100:
                                is_ratios.append(math.exp(100))
                                print('large is_ratios_found')
                            else:
                                is_ratios.append(math.exp(-proposal_log_probs[j] - mis_logits[j].item()))
                        # (without cache)
                        max = is_ratios[0]
                        current = 0
                        lengths = 0
                        for j in range(sample_num):
                            if is_ratios[j]> max:
                                mis_results[current] = lengths
                                max = is_ratios[j]
                                current = j
                                lengths = 1
                            else:
                                lengths += 1
                        mis_results[current] = lengths
                            # sample should be added
                        normalize = sum(mis_results[tmp]*is_ratios[tmp] for tmp in mis_results)
                        if normalize>0:
                            for index, length in mis_results.items():
                                if proposal_wrong_num[index] in statistic:
                                    statistic[proposal_wrong_num[index]] += mis_logits[index].item()
                                    statistic_count[proposal_wrong_num[index]] += 1
                                else:
                                    statistic[5] += mis_logits[index].item()
                                    statistic_count[5] += 1
                                #if is_ratios[j]/normalize>0.003: # add reject control
                                loss = loss - (is_ratios[index]*length* mis_logits[index])/normalize
                                #else:
                                #if random.random()<is_ratios[j]*200/normalize:
                                #loss = loss - 0.005*mis_logits[j]
                if loss!=0.0 and (not torch.isnan(loss).any()) and (not torch.isinf(loss).any()):
                    loss.backward()
                    epoch_loss += loss.item()
                else:
                    print(loss)
                    nan_num +=1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory, batch size: {}".format(cfg.batch_size))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        tmp = [statistic[i]/statistic_count[i] for i in statistic]
        print(tmp)
        print(epoch_loss / num_batches)
        print(gt_energy / num_batches)
        # Evaluate and save checkpoint
        score, precision, recall, f1 = rescore(proposal_model, model, test_dataloader, cfg, tokenizer) # dev_dataloader
        metrics_to_log["eval_score"] = score
        logging.info("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        s = score + recall + f1
        print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        #if (epoch-1)%5==0:
        #    score, precision, recall, f1 = rescore(proposal_model, model, train_dataloader, cfg, tokenizer)
        #    print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        if s > best_score:
            print("best checkpoints saved, score:{s}")
            #logging.info("New best results found! Score: {}".format(best_score))
            if not os.path.exists(cfg.ebm_save_path):
                os.mkdir(cfg.ebm_save_path)
            tokenizer.save_pretrained(cfg.ebm_save_path)
            torch.save(model.state_dict(), os.path.join(cfg.ebm_save_path, "model_mis.pt"))
            best_score = s
        if nan_num>100:
            break
    #model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "model.pt")))
    #score = evaluate(model, test_dataloader, cfg)
    #print(score)
    return

def rescore(proposal_model, model, val_dataloader, hparams, tokenizer): # tokenizer
    origin_threshold = 0.5 # means the sampling probability increased, discard used in sampling
    accept_threshold = 0.5 # used to simplify computation, accept those triples above the Threshold and ignore them
    reject_threshold = 0.0 # used to simplify computation, discard those triples above the Threshold and ignore them
    topk_num = cfg.test_num
    model.eval()
    joint_acc = 0
    joint_acc_e = 0
    joint_acc_ea = 0
    total_case = 0
    joint_acc_a = 0
    total_case_a = 0
    total_case_e = 0
    num_batches = 0
    global_step = 0
    tp_a = 0
    fp_a = 0
    fn_a = 0
    tp = 0
    fp = 0
    fn = 0
    labels = []
    predicts = []
    with torch.no_grad():
        for batch in val_dataloader:
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                if cfg.only_one_model:
                    #logits = proposal_model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits # ,labels=batch["label"]
                    #probs = F.softmax(logits, dim=1)
                    #accs = ((batch["label"]==0)==(probs[:,0]>threshold)).cpu().tolist()
                    #labels.extend((batch["label"]==0).cpu().tolist())
                    #predicts.extend((probs[:,0]>threshold).cpu().tolist())
                    #acc.extend(accs) 
                    for i in range(len(batch)):
                        # get_proposal_prob
                        p_batch = collate_fn(batch['cases'][i])
                        for key, val in p_batch.items():
                            if type(p_batch[key]) is list:
                                continue
                            p_batch[key] = p_batch[key].to(cfg.device[-1])
                        if p_batch!={}:
                            p_logits = proposal_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                            probs = F.softmax(p_logits, dim=1)
                            accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                        else:
                            accept_prob = []
                        #Threshold
                        triple_num = len(accept_prob)
                        triples_accepted = []
                        triples_accepted_idx = []
                        accepted_probs = 1.0
                        triples = []
                        triple_probs = []
                        triple_idxs = []
                        proposals = [] # use proposals to indicate the possible results
                        org_proposals = []
                        gt = []
                        accept_result = []

                        # change propose to topk
                        for num in range(triple_num):
                            if batch['cases'][i][num]['label'] == 1:
                                gt.append(num)
                            if accept_prob[num] > origin_threshold :
                                accept_result.append(num)
                            if accept_prob[num]> accept_threshold:
                                triples_accepted.append(tokenizer.decode(batch['cases'][i][num]['triple']).replace('[CLS]','').replace('[SEP]','')) 
                                accepted_probs = accepted_probs*accept_prob[num] 
                                triples_accepted_idx.append(num)
                            elif accept_prob[num]> reject_threshold:
                                triples.append(tokenizer.decode(batch['cases'][i][num]['triple']).replace('[CLS]','').replace('[SEP]',''))
                                triple_probs.append(accept_prob[num])
                                triple_idxs.append(num)
                        proposals = [(triples_accepted, accepted_probs, triples_accepted_idx)]
                        # topk_num, get_topk here by using beam search
                        for t_num in range(len(triples)):
                            triple = triples[t_num]
                            triple_prob = triple_probs[t_num]
                            triple_idx = triple_idxs[t_num]
                            new_proposals = [] # a temp variable to store the iterated proposal
                            for proposal in proposals:
                                new_proposals.append((proposal[0], proposal[1]*(1-triple_prob), proposal[2]))
                                tmp = copy.deepcopy(proposal)
                                tmp[0].append(triple)
                                #tmp[1] = proposal[1]*triple_prob
                                tmp[2].append(triple_idx)
                                new_proposals.append((tmp[0], proposal[1]*triple_prob, tmp[2]))
                            if len(new_proposals)>topk_num:
                                new_proposals.sort(key=lambda x:x[1], reverse=True)
                                proposals = copy.deepcopy(new_proposals[:topk_num])
                            else:
                                proposals = copy.deepcopy(new_proposals)
                        proposals.sort(key=lambda x:x[1], reverse=True)
                        topk = proposals[:topk_num]
                        #result = sorted(data,key=lambda x:(x[0],x[1].lower()))
                        # proposals.append('；'.join(proposal))
                        # proposed_probs.append(math.log(proposed_prob))
                        #org_proposals.append(org_proposal)
                        """
                        proposed_probs = []
                        for sample_num in range(cfg.train_sample_num): # cfg.test_num
                            proposal = []
                            org_proposal = []
                            gt = []
                            accept_result = []
                            random.random()
                            proposed_prob = 1.0
                            for num in range(triple_num):
                                p = random.random()
                                if p < accept_prob[num] + rescore_threshold:
                                    proposal.append(tokenizer.decode(batch['cases'][i][num]['triple']).replace('[CLS]','').replace('[SEP]','')) 
                                    org_proposal.append(num) # num batch['cases'][i][num]
                                    proposed_prob = proposed_prob*accept_prob[num]
                                else:
                                    proposed_prob = proposed_prob*(1-accept_prob[num])
                                if batch['cases'][i][num]['label'] == 1:
                                    gt.append(num)
                                if accept_prob[num] >(0.5 - rescore_threshold) :
                                    accept_result.append(num)
                                    # can directly concatenate all the triples to improve efficiency
                            proposals.append('；'.join(proposal))
                            proposed_probs.append(math.log(proposed_prob))
                            org_proposals.append(org_proposal)
                        """
                        input = get_retrieval_sequence(tokenizer, [batch['context'][i]]*len(topk), ['；'.join(item[0]) for item in topk])
                        input.to(cfg.device[0])
                        #if cfg.add_extra_feature:
                        positive_count = torch.tensor([float(len(item[0])) for item in topk], dtype=torch.float).to(cfg.device[0])
                        if cfg.add_extra_feature:
                            logits = model(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count).to('cpu').tolist()
                        else:
                            logits = model(input_ids=input['input_ids'], attention_mask=input["attention_mask"]).to('cpu').tolist()
                        if cfg.residual:
                            for j in range(len(logits)):
                                logits[j] = logits[j][0] + math.log(topk[j][1])
                        final = logits.index(max(logits))
                        triples = proposals[final][2]

                        # compute acc
                        total_case_e = total_case_e + 1
                        if gt!=[] and accept_result!=[]:
                            total_case_a = total_case_a + 1
                            joint_acc_a += set(accept_result)==set(gt)
                            joint_acc_ea += set(accept_result)==set(gt)
                        else:
                            joint_acc_ea += 1
                        for num in range(triple_num):
                            tp_a += (num in accept_result) and (num in gt)
                            fp_a += (num in accept_result) and (num not in gt)
                            fn_a += (num not in accept_result) and (num in gt)
                        if gt!=[] and triples!=[]:
                            total_case = total_case + 1
                            joint_acc += set(triples)==set(gt)
                            joint_acc_e += set(triples)==set(gt)
                        else:
                            joint_acc_e += 1
                        for num in range(triple_num):
                            tp += (num in triples) and (num in gt)
                            fp += (num in triples) and (num not in gt)
                            fn += (num not in triples) and (num in gt)
                #else:
                #    _,batch_acc,predict,label = model(input_sent=batch["context"], attention_sent=batch["context_attention"],input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])
                #    labels.extend(label)
                #    predicts.extend(predict)
                #    acc.append(batch_acc)
    # the positive and negative are assigned to 0 and 1 respectively, so we should calculate the f1 of 0, dealt with by getting the probs of 0
    recall = tp/(tp+fn)
    precision= tp/(tp+fp)
    f1 = 2*precision*recall/(precision + recall+0.000001)
    recall_a = tp_a/(tp_a+fn_a)
    precision_a= tp_a/(tp_a+fp_a)
    f1_a = 2*precision_a*recall_a/(precision_a + recall_a+0.000001)
    print("j_acc with empty slot: energy {}, proposal: {}".format(joint_acc_e/total_case_e, joint_acc_ea/total_case_e))
    print("proposal result: score: {}, precision: {}, recall: {}, f1: {}".format(joint_acc_a/total_case_a, precision_a, recall_a, f1_a))
    return joint_acc/total_case, precision, recall, f1

def evaluate(model, val_dataloader, hparams): # tokenizer
    threshold = 0.5
    model.eval()
    acc = []
    num_batches = 0
    global_step = 0
    labels = []
    predicts = []
    with torch.no_grad():
        for batch in val_dataloader:
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(hparams.device[0])
                if cfg.only_one_model:
                    logits = model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits # ,labels=batch["label"]
                    probs = F.softmax(logits, dim=1)
                    accs = ((batch["label"]==0)==(probs[:,0]>threshold)).cpu().tolist()
                    labels.extend((batch["label"]==0).cpu().tolist())
                    predicts.extend((probs[:,0]>threshold).cpu().tolist())
                    acc.extend(accs) 
                else:
                    _,batch_acc,predict,label = model(input_sent=batch["context"], attention_sent=batch["context_attention"],input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])
                    labels.extend(label)
                    predicts.extend(predict)
                    acc.append(batch_acc)
    # the positive and negative are assigned to 0 and 1 respectively, so we should calculate the f1 of 0
    positive = sum(labels)
    retrieved = sum((predicts[i] and labels[i]) for i in range(len(predicts)))
    predicted = sum(predicts)
    recall = retrieved/positive
    precision= retrieved/predicted
    f1 = 2*precision*recall/(precision + recall)
    return sum(acc)/len(acc), precision, recall, f1

def test(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_save_path)
    if cfg.only_one_model:
        model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
        model.to(cfg.device[0])
    else:
        model = EncoderModel(cfg,tokenizer)
        model.load_state_dict(torch.load(os.path.join(cfg.retrieval_save_path, "model.pt")))

    encoded_data = read_data(tokenizer, retrieve=True)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    acc, precision, recall, f1 = evaluate(model, test_dataloader, cfg)
    print("acc: {}, precision: {}, recall: {}, f1: {}".format(acc, precision, recall, f1))
    return recall

if __name__ == "__main__":
    # mode settings
    cfg.only_one_model = True
    cfg.debugging = False
    #cfg.train_retrieve = False
    test_retrieve = False
    cfg.add_extra_feature = False
    if cfg.add_extra_feature:
        cfg.reg_weight = 15.0
    # ebm
    cfg.train_ebm = False
    if (not cfg.train_ebm) and (not cfg.train_retrieve):
        test_ebm = True
    else:
        test_ebm = False
    cfg.use_all_proposal = True
    # mis settings
    cfg.train_ebm_mis = True # modified, MIS or not
    if cfg.train_ebm_mis:
        cfg.use_all_proposal = True # mis uses all proposals

    # other settings    
    cfg.residual = True # residual or not
    cfg.reject_control = False # reject_control or not
    #cfg.device = [6, 7]
    if cfg.train_ebm:
        cfg.batch_size = 6
        cfg.eval_batch_size = 4
        train_ebm(cfg)
    """
    if cfg.train_ebm_mis:
        cfg.batch_size = 6
        cfg.eval_batch_size = 4
        train_ebm_with_mis(cfg)
    """
    if test_ebm:
        cfg.test_num = 16
        cfg.eval_batch_size = 8
        tokenizer = BertTokenizer.from_pretrained(cfg.ebm_save_path)
        model = EBM(cfg,tokenizer)
        save_path = os.path.join(cfg.ebm_save_path, 
        f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "") + ".pt")
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        model.to(cfg.device[0])
        if cfg.only_one_model:
            proposal_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
            proposal_model.to(cfg.device[-1])
        encoded_data = read_data(tokenizer, ebm=True)
        test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
        score, precision, recall, f1 = rescore(proposal_model, model, test_dataloader, cfg, tokenizer)
        print(save_path)
        print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
    if cfg.train_retrieve:
        train(cfg)
    if test_retrieve:
       test(cfg)
