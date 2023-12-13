import copy
from abc import ABC

from transformers import T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.utils.rnn import pad_sequence
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.generation_utils import GenerationMixin
from torch import nn, Tensor
import torch.distributed as dist
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from utils.io import read_pkl, write_pkl, read_file
from collections import defaultdict
from copy import deepcopy
import numpy as np
import json
import faiss
import torch
import os
import argparse
import time
from tqdm import tqdm
import torch


class Tree:
    def __init__(self):
        self.root = dict()

    def set(self, path):
        pointer = self.root
        for i in path:
            if i not in pointer:
                pointer[i] = dict()
            pointer = pointer[i]

    def set_all(self, path_list):
        for path in tqdm(path_list):
            self.set(path)

    def find(self, path):
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        pointer = self.root
        for i in path:
            if i not in pointer:
                return []
            pointer = pointer[i]
        return list(pointer.keys())

    def __call__(self, batch_id, path):
        return self.find(path)


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    all_dense_embed: Optional[torch.FloatTensor] = None
    continuous_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None
    probability: Optional[torch.FloatTensor] = None
    code_logits: Optional[torch.FloatTensor] = None


@torch.no_grad()
def sinkhorn_algorithm(out: Tensor, epsilon: float,
                       sinkhorn_iterations: int,
                       use_distrib_train: bool):
    Q = torch.exp(out / epsilon)  # Q is M-K-by-B

    M = Q.shape[0]
    B = Q.shape[2]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    if use_distrib_train:
        B *= dist.get_world_size()
        dist.all_reduce(sum_Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        if use_distrib_train:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q


@torch.no_grad()
def sinkhorn_raw(out: Tensor, epsilon: float,
                 sinkhorn_iterations: int,
                 use_distrib_train: bool):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper

    B = Q.shape[1]
    K = Q.shape[0]  # how many prototypes
    # make the matrix sums to 1
    sum_Q = torch.clamp(torch.sum(Q), min=1e-5)
    if use_distrib_train:
        B *= dist.get_world_size()
        dist.all_reduce(sum_Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.clamp(torch.sum(Q, dim=1, keepdim=True), min=1e-5)
        if use_distrib_train:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.clamp(torch.sum(torch.sum(Q, dim=0, keepdim=True), dim=1, keepdim=True), min=1e-5)
        Q /= B
    Q *= B
    return Q.t()


def get_optimizer(model, lr):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and "centroids" not in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       n not in decay_parameters and "centroids" not in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "centroids" in n],
            "weight_decay": 0.0,
            'lr': lr * 20
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr)
    return optimizer


class Model(nn.Module, GenerationMixin, ABC):
    def __init__(self, model, use_constraint: bool, sk_epsilon: float = 0.03, sk_iters: int = 100, code_length=1,
                 zero_inp=False, code_number=10):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.can_generate = lambda: True
        hidden_size = model.config.hidden_size

        self.use_constraint, self.sk_epsilon, self.sk_iters = use_constraint, sk_epsilon, sk_iters

        # Codebook of each time step
        self.centroids = nn.ModuleList([nn.Linear(hidden_size, code_number, bias=False) for _ in range(code_length)])
        self.centroids.requires_grad_(True)

        # Code embedding (input to the decoder)
        self.code_embedding = nn.ModuleList([nn.Embedding(code_number, hidden_size) for _ in range(code_length)])
        self.code_embedding.requires_grad_(True)

        self.code_length = code_length
        self.zero_inp = zero_inp
        self.code_number = code_number

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    @torch.no_grad()
    def quantize(self, probability, use_constraint=None):
        # batchsize_per_device = len(continuous_embeds)
        # distances = ((continuous_embeds.reshape(batchsize_per_device, self.config.MCQ_M, 1, -1).transpose(0,1) -
        #               self.centroids.unsqueeze(1)) ** 2).sum(-1)  # M, bs', K
        distances = -probability
        use_constraint = self.use_constraint if use_constraint is None else use_constraint
        # raw_code = torch.argmin(distances, dim=-1)
        # print('In', torch.argmin(distances, dim=-1))
        if not use_constraint:
            codes = torch.argmin(distances, dim=-1)  # M, bs
        else:
            distances = self.center_distance_for_constraint(distances)  # to stablize
            # avoid nan
            distances = distances.double()
            # Q = sinkhorn_algorithm(
            #     -distances.transpose(1, 2),
            #     self.sk_epsilon,
            #     self.sk_iters,
            #     use_distrib_train=dist.is_initialized()
            # ).transpose(1, 2)  # M-B-K
            Q = sinkhorn_raw(
                -distances,
                self.sk_epsilon,
                self.sk_iters,
                use_distrib_train=dist.is_initialized()
            )  # B-K
            codes = torch.argmax(Q, dim=-1)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
        # print('Out', codes)
        # print('Equal', (raw_code == codes).float().mean())
        # codes = codes.t()  # bs, M
        # input('>')
        return codes

    def decode(self, codes, centroids=None):
        M = codes.shape[1]
        if centroids is None:
            centroids = self.centroids
        if isinstance(codes, torch.Tensor):
            assert isinstance(centroids, torch.Tensor)
            first_indices = torch.arange(M).to(codes.device)
            first_indices = first_indices.expand(*codes.shape).reshape(-1)
            quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
        elif isinstance(codes, np.ndarray):
            if isinstance(centroids, torch.Tensor):
                centroids = centroids.detach().cpu().numpy()
            first_indices = np.arange(M)
            first_indices = np.tile(first_indices, len(codes))
            quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
        else:
            raise NotImplementedError()
        return quant_embeds

    def embed_decode(self, codes, centroids=None):
        if centroids is None:
            centroids = self.centroids[-1]
        quant_embeds = F.embedding(codes, centroids.weight)
        return quant_embeds

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: M, bs, K
        max_distance = distances.max()
        min_distance = distances.min()
        if dist.is_initialized():
            dist.all_reduce(max_distance, torch.distributed.ReduceOp.MAX)
            dist.all_reduce(min_distance, torch.distributed.ReduceOp.MIN)
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert torch.all(amplitude > 0)
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, aux_ids=None, return_code=False,
                return_quantized_embedding=False, use_constraint=None, encoder_outputs=None, **kwargs):
        if decoder_input_ids is None or self.zero_inp:
            decoder_input_ids = torch.zeros(input_ids.size(0), self.code_length).long().to(input_ids.device)

        # decoder_inputs_embeds = self.code_embedding(decoder_input_ids)

        decoder_inputs_embeds = []
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            code_embedding = self.code_embedding[i]
            decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )
        decoder_outputs = model_outputs.decoder_hidden_states[-1]
        all_dense_embed = decoder_outputs.view(decoder_outputs.size(0), -1).contiguous()
        dense_embed = decoder_outputs[:, -1].contiguous()

        code_logits = []
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            centroid = self.centroids[i]
            code_logits.append(centroid(decoder_outputs[:, i]))
        code_logits = torch.stack(code_logits, dim=1)
        # code_logits = self.centroids(decoder_outputs)

        probability = code_logits[:, -1].contiguous()
        # probability = torch.mm(dense_embed, self.centroids.transpose(0, 1))
        discrete_codes = self.quantize(probability, use_constraint=use_constraint)

        if aux_ids is None:
            aux_ids = discrete_codes

        quantized_embeds = self.embed_decode(aux_ids) if return_quantized_embedding else None

        if self.code_length == 1:
            return_code_logits = None
        else:
            return_code_logits = code_logits[:, :-1].contiguous()

        quant_output = QuantizeOutput(
            logits=code_logits,
            all_dense_embed=all_dense_embed,
            continuous_embeds=dense_embed,
            quantized_embeds=quantized_embeds,
            discrete_codes=discrete_codes,
            probability=probability,
            code_logits=return_code_logits,
        )
        return quant_output


class OurTrainer:
    @staticmethod
    def _gather_tensor(t: Tensor, local_rank):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[local_rank] = t
        return all_tensors

    @staticmethod
    def gather_tensors(t: Tensor, local_rank=None):
        if local_rank is None:
            local_rank = dist.get_rank()
        return torch.cat(OurTrainer._gather_tensor(t, local_rank))

    @staticmethod
    @torch.no_grad()
    def test_step(model: Model, batch, use_constraint=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'],
                                              aux_ids=None, return_code=False,
                                              return_quantized_embedding=False, use_constraint=use_constraint)
        doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
                                            decoder_input_ids=batch['ids'],
                                            aux_ids=None, return_code=False,
                                            return_quantized_embedding=False, use_constraint=use_constraint)
        return query_outputs, doc_outputs

    @staticmethod
    def simple_train_step(model: Model, batch, gathered=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'])
        # doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
        #                                     decoder_input_ids=batch['ids'])

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            code_number = model.module.code_number
        else:
            code_number = model.code_number
        # code_number = 10
        query_code_loss = F.cross_entropy(query_outputs.code_logits.view(-1, code_number),
                                          batch['ids'][:, 1:].reshape(-1))
        # doc_code_loss = F.cross_entropy(doc_outputs.code_logits.view(-1, code_number),
        #                                 batch['ids'][:, 1:].reshape(-1))
        query_prob = query_outputs.probability
        aux_query_code_loss = F.cross_entropy(query_prob, batch['aux_ids'])
        code_loss = query_code_loss
        return dict(
            loss=query_code_loss + aux_query_code_loss,
        )

    @staticmethod
    def train_step(model: Model, batch, gathered=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'],
                                              aux_ids=batch['aux_ids'], return_code=True,
                                              return_quantized_embedding=True)
        doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
                                            decoder_input_ids=batch['ids'],
                                            aux_ids=batch['aux_ids'], return_code=True,
                                            return_quantized_embedding=True)
        query_embeds = query_outputs.continuous_embeds
        doc_embeds = doc_outputs.continuous_embeds
        codes_doc = doc_outputs.discrete_codes
        quant_doc_embeds = doc_outputs.quantized_embeds
        query_prob = query_outputs.probability
        doc_prob = doc_outputs.probability

        query_all_embeds = query_outputs.all_dense_embed
        doc_all_embeds = doc_outputs.all_dense_embed

        if gathered is None:
            gathered = dist.is_initialized()

        cl_loss = OurTrainer.compute_contrastive_loss(query_embeds, doc_embeds, gathered=False)  # retrieval

        all_cl_loss = OurTrainer.compute_contrastive_loss(query_all_embeds, doc_all_embeds,
                                                          gathered=dist.is_initialized())  # retrieval (used when dist)

        # cl_d_loss = OurTrainer.compute_contrastive_loss(doc_embeds, query_embeds, gathered=gathered)
        # cl_loss = cl_q_loss + cl_d_loss

        # mse_loss = 0
        cl_dd_loss = OurTrainer.compute_contrastive_loss(
            quant_doc_embeds + doc_embeds - doc_embeds.detach(), doc_embeds.detach(), gathered=False)  # reconstruction
        # mse_loss = ((quant_doc_embeds - doc_embeds.detach()) ** 2).sum(-1).mean()

        # codes_doc_cpu = codes_doc.cpu().tolist()
        # print(balance(codes_doc_cpu))
        # print(codes_doc)
        query_ce_loss = F.cross_entropy(query_prob, codes_doc.detach())  # commitment
        doc_ce_loss = F.cross_entropy(doc_prob, codes_doc.detach())  # commitment
        ce_loss = query_ce_loss + doc_ce_loss  # commitment

        code_loss = 0
        if query_outputs.code_logits is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                code_number = model.module.code_number
            else:
                code_number = model.code_number
            query_code_loss = F.cross_entropy(query_outputs.code_logits.view(-1, code_number),
                                              batch['ids'][:, 1:].reshape(-1))
            doc_code_loss = F.cross_entropy(doc_outputs.code_logits.view(-1, code_number),
                                            batch['ids'][:, 1:].reshape(-1))
            code_loss = query_code_loss + doc_code_loss  # commitment
        if batch['aux_ids'] is not None:
            aux_query_code_loss = F.cross_entropy(query_prob, batch['aux_ids'])
            aux_doc_code_loss = F.cross_entropy(doc_prob, batch['aux_ids'])
            aux_code_loss = aux_query_code_loss + aux_doc_code_loss  # commitment on last token
            # print('Q', aux_query_code_loss.item(), 'D', aux_doc_code_loss.item())
            if aux_code_loss.isnan():
                aux_code_loss = 0
        else:
            aux_code_loss = 0

        if dist.is_initialized():
            all_doc_embeds = OurTrainer.gather_tensors(doc_embeds)
            global_mean_doc_embeds = all_doc_embeds.mean(dim=0)
            local_mean_doc_embeds = doc_embeds.mean(dim=0)
            clb_loss = F.mse_loss(local_mean_doc_embeds, global_mean_doc_embeds.detach())  # not used
        else:
            clb_loss = 0

        return dict(
            cl_loss=cl_loss,
            all_cl_loss=all_cl_loss,
            mse_loss=0,
            ce_loss=ce_loss,
            code_loss=code_loss,
            aux_code_loss=aux_code_loss,
            cl_dd_loss=cl_dd_loss,
            clb_loss=clb_loss
        )

    @staticmethod
    def compute_contrastive_loss(query_embeds, doc_embeds, gathered=True):
        if gathered:
            gathered_query_embeds = OurTrainer.gather_tensors(query_embeds)
            gathered_doc_embeds = OurTrainer.gather_tensors(doc_embeds)
        else:
            gathered_query_embeds = query_embeds
            gathered_doc_embeds = doc_embeds
        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_doc_embeds.transpose(0, 1))
        # similarities = similarities
        co_loss = F.cross_entropy(similarities, labels)
        return co_loss


class BiDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_doc_len=512, max_q_len=128, ids=None, batch_size=1, aux_ids=None):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_doc_len = max_doc_len
        self.max_q_len = max_q_len
        self.ids = ids
        self.batch_size = batch_size

        if self.batch_size != 1:
            ids_to_item = defaultdict(list)
            for i, item in enumerate(self.data):
                ids_to_item[str(ids[item[1]])].append(i)
            for key in ids_to_item:
                np.random.shuffle(ids_to_item[key])
            self.ids_to_item = ids_to_item
        else:
            self.ids_to_item = None
        self.aux_ids = aux_ids

    def getitem(self, item):
        queries, doc_id = self.data[item]
        if isinstance(queries, list):
            query = np.random.choice(queries)
        else:
            query = queries

        while isinstance(doc_id, list):
            doc_id = doc_id[0]

        doc = self.corpus[doc_id]
        if self.ids is None:
            ids = [0]
        else:
            ids = self.ids[doc_id]
        if self.aux_ids is None:
            aux_ids = -100
        else:
            aux_ids = self.aux_ids[doc_id]
        return (torch.tensor(self.tokenizer.encode(query, truncation=True, max_length=self.max_q_len)),
                torch.tensor(self.tokenizer.encode(doc, truncation=True, max_length=self.max_doc_len)),
                ids, aux_ids)

    def __getitem__(self, item):
        if self.batch_size == 1:
            return [self.getitem(item)]
        else:
            # item_set = self.ids_to_item[str(self.ids[self.data[item][1]])]
            # new_item_set = [item] + [i for i in item_set if i != item]
            # work_item_set = new_item_set[:self.batch_size]
            # new_item_set = new_item_set[self.batch_size:] + work_item_set
            # self.ids_to_item[str(self.ids[self.data[item][1]])] = new_item_set

            item_set = deepcopy(self.ids_to_item[str(self.ids[self.data[item][1]])])
            np.random.shuffle(item_set)
            item_set = [item] + [i for i in item_set if i != item]
            work_item_set = item_set[:self.batch_size]

            if len(work_item_set) < self.batch_size:
                rand_item_set = np.random.randint(len(self), size=self.batch_size * 2)
                rand_item_set = [i for i in rand_item_set if i != item]
                work_item_set = work_item_set + rand_item_set
                work_item_set = work_item_set[:self.batch_size]

            collect = []
            for item in work_item_set:
                query, doc, ids, aux_ids = self.getitem(item)
                collect.append((query, doc, ids, aux_ids))
            return collect

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        data = sum(data, [])
        query, doc, ids, aux_ids = zip(*data)
        query = pad_sequence(query, batch_first=True, padding_value=0)
        doc = pad_sequence(doc, batch_first=True, padding_value=0)
        ids = torch.tensor(ids)
        if self.aux_ids is None:
            aux_ids = None
        else:
            aux_ids = torch.tensor(aux_ids)
        return {
            'query': query,
            'doc': doc,
            'ids': ids,
            'aux_ids': aux_ids
        }


def safe_load(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    # size_not_match = [k for k,v in state_dict.items() if model_state_dict_keys[k]]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)
    model.load_state_dict(state_dict, strict=False)


def safe_load_embedding(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)

    matched_state_dict = deepcopy(model.state_dict())
    for key in model_state_dict_keys:
        if key in state_dict:
            file_size = state_dict[key].size(0)
            model_embedding = matched_state_dict[key].clone()
            model_size = model_embedding.size(0)
            model_embedding[:file_size, :] = state_dict[key][:model_size, :]
            matched_state_dict[key] = model_embedding
            print(f'Copy {key} {matched_state_dict[key].size()} from {state_dict[key].size()}')
    model.load_state_dict(matched_state_dict, strict=False)


def safe_save(accelerator, model, save_path, epoch, end_epoch=100, save_step=1, last_checkpoint=None):
    os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process and epoch < end_epoch and epoch % save_step == 0:
        unwrap_model = accelerator.unwrap_model(model)
        accelerator.save(unwrap_model.state_dict(), f'{save_path}/{epoch}.pt')
        accelerator.save(unwrap_model.model.state_dict(), f'{save_path}/{epoch}.pt.model')
        accelerator.save(unwrap_model.centroids.state_dict(), f'{save_path}/{epoch}.pt.centroids')
        accelerator.save(unwrap_model.code_embedding.state_dict(), f'{save_path}/{epoch}.pt.embedding')
        accelerator.print(f'Save model {save_path}/{epoch}.pt')
        last_checkpoint = f'{save_path}/{epoch}.pt'
    return epoch + 1, last_checkpoint


def simple_loader(data, corpus, tokenizer, ids, aux_ids, accelerator):
    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer,
                        max_doc_len=128, max_q_len=32, ids=ids, batch_size=1, aux_ids=aux_ids)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=32,
                                              shuffle=True, num_workers=4)
    data_loader = accelerator.prepare(data_loader)
    return data_loader


def train(config):
    accelerator = Accelerator(gradient_accumulation_steps=1)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_model = config.get('prev_model', None)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)

    train_data = config.get('train_data', 'dataset/nq320k/train.json')
    corpus_data = config.get('corpus_data', 'dataset/nq320k/corpus_lite.json')
    epochs = config.get('epochs', 100)
    in_batch_size = config.get('batch_size', 128)
    end_epoch = epochs

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    save_step = 1
    batch_size = 1
    lr = 5e-4

    accelerator.print(save_path)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = Model(model=t5, code_length=code_length,
                  use_constraint=True, sk_epsilon=1, zero_inp=False, code_number=code_num)

    if prev_model is not None:
        safe_load(model.model, f'{prev_model}.model')
        safe_load(model.centroids, f'{prev_model}.centroids')
        safe_load_embedding(model.code_embedding, f'{prev_model}.embedding')

    if config.get('codebook_init', None) is not None:
        model.centroids[0].weight.data = torch.tensor(read_pkl(config.get('codebook_init')))

    for i in range(code_length - 1):
        model.centroids[i].requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    data = json.load(open(train_data))
    data.extend(json.load(open(f'{train_data}.qg.json')))
    corpus = json.load(open(corpus_data))

    grouped_data = defaultdict(list)
    for i, item in enumerate(data):
        query, docid = item
        if isinstance(docid, list):
            docid = docid[0]
        grouped_data[docid].append(query)
    data = [[query_list, docid] for docid, query_list in grouped_data.items()]

    ids, aux_ids = None, None

    if prev_id is not None:
        ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        ids = [[0]] * len(corpus)

    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer,
                        max_doc_len=128, max_q_len=32, ids=ids, batch_size=in_batch_size, aux_ids=aux_ids)
    accelerator.print(f'data size={len(dataset)}')

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    optimizer = AdamW(model.parameters(), lr)
    # optimizer = get_optimizer(model, lr=lr)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_constant_schedule(optimizer)

    w_1 = {'cl_loss': 0.5, 'all_cl_loss': 0, 'ce_loss': 0, 'code_loss': 0.5, 'aux_code_loss': 0, 'mse_loss': 0,
           'cl_dd_loss': 0, 'clb_loss': 0}
    w_2 = {'cl_loss': 0.5, 'all_cl_loss': 0, 'ce_loss': 0.5, 'code_loss': 0.5, 'aux_code_loss': 0, 'mse_loss': 0,
           'cl_dd_loss': 0.1, 'clb_loss': 0}
    w_3 = {'cl_loss': 0, 'all_cl_loss': 0, 'ce_loss': 0.5, 'code_loss': 1, 'aux_code_loss': 0, 'mse_loss': 0,
           'cl_dd_loss': 0, 'clb_loss': 0}
    loss_w = [None, w_1, w_2, w_3][config['loss_w']]

    step, epoch = 0, 0
    epoch_step = len(data_loader) // in_batch_size
    # safe_save(accelerator, model, save_path, -1, end_epoch=end_epoch)
    last_checkpoint = None

    for _ in range(epochs):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            step += 1
            with accelerator.accumulate(model):
                # losses = OurTrainer.train_step(model, batch, gathered=False)
                # loss = sum([v * loss_w[k] for k, v in losses.items()])
                # accelerator.backward(loss)
                # accelerator.clip_grad_norm_(model.parameters(), 1.)
                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()
                #
                # loss = accelerator.gather(loss).mean().item()
                loss = 0.
                loss_report.append(loss)
                tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

                if in_batch_size != 1 and step > (epoch + 1) * epoch_step:
                    epoch, last_checkpoint = safe_save(accelerator, model, save_path, epoch, end_epoch=end_epoch,
                                                       save_step=save_step,
                                                       last_checkpoint=last_checkpoint)
                if epoch >= end_epoch:
                    break
        if in_batch_size == 1:
            epoch = safe_save(accelerator, model, save_path, epoch, end_epoch=end_epoch, save_step=save_step)

    return last_checkpoint


def balance(code, prefix=None, ncentroids=10):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        prefix_code = defaultdict(list)
        for c, p in zip(code, prefix):
            prefix_code[p].append(c)
        scores = []
        for p, p_code in prefix_code.items():
            scores.append(balance(p_code, ncentroids=ncentroids))
        return {'Avg': sum(scores) / len(scores), 'Max': max(scores), 'Min': min(scores), 'Flat': balance(code)}
    num = [code.count(i) for i in range(ncentroids)]
    base = len(code) // ncentroids
    move_score = sum([abs(j - base) for j in num])
    score = 1 - move_score / len(code) / 2
    return score


def conflict(code, prefix=None):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        code = [f'{p}{c}' for c, p in zip(code, prefix)]
    code = [str(c) for c in code]
    freq_count = defaultdict(int)
    for c in code:
        freq_count[c] += 1
    max_value = max(list(freq_count.values()))
    min_value = min(list(freq_count.values()))
    len_set = len(set(code))
    return {'Max': max_value, 'Min': min_value, 'Type': len_set, '%': len_set / len(code)}


def ress(code, prefix=None):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        code = [f'{p}{c}' for c, p in zip(code, prefix)]
    freq_count = defaultdict(int)
    for c in code:
        freq_count[c] += 1
    freq_count = [y for x, y in freq_count.items()]
    freq_count.sort()
    return freq_count


def ress_by_prefix(code, prefix=None):
    freq_count = defaultdict(list)
    for c, p in zip(code, prefix):
        p = str(p)
        freq_count[p].append(c)
    freq_count = [[len(v), len(set(v))] for k, v in freq_count.items()]
    freq_count.sort(key=lambda x: x[1])
    return freq_count


def test(config):
    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)
    batch_size = 32
    epochs = config.get('epochs', 100)

    dev_data = config.get('dev_data', config.get('dev_data'))
    corpus_data = config.get('corpus_data', config.get('corpus_data'))

    data = json.load(open(dev_data))
    corpus = json.load(open(corpus_data))

    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = Model(model=t5, use_constraint=False, code_length=code_length, zero_inp=False, code_number=code_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ids = None
    if prev_id is not None:
        corpus_ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        corpus_ids = [[0]] * len(corpus)
    aux_ids = None

    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=corpus_ids,
                        aux_ids=aux_ids)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    model = model.cuda()
    model.eval()

    seen_split = json.load(open(f'{dev_data}.seen'))
    unseen_split = json.load(open(f'{dev_data}.unseen'))

    for epoch in range(epochs):
        if not os.path.exists(f'{save_path}/{epoch}.pt'):
            continue
        print(f'Test {save_path}/{epoch}.pt')

        corpus_ids = [[0, *line] for line in json.load(open(f'{save_path}/{epoch}.pt.code'))]
        safe_load(model, f'{save_path}/{epoch}.pt')
        tree = Tree()
        tree.set_all(corpus_ids)

        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        output_all = []
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items() if v is not None}
                top_k = 10
                output = model.generate(
                    input_ids=batch['query'].cuda(),
                    attention_mask=batch['query'].ne(0).cuda(),
                    max_length=code_length + 1,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=tree
                )
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line.cpu().tolist())
                new_output.append(beam)
                output_all.extend(new_output)

        query_ids = [x[1] for x in data]

        docid_to_doc = defaultdict(list)
        for i, item in enumerate(corpus_ids):
            docid_to_doc[str(item)].append(i)
        predictions = []
        for line in output_all:
            new_line = []
            for s in line:
                s = str(s)
                if s not in docid_to_doc:
                    continue
                tmp = docid_to_doc[s]
                # np.random.shuffle(tmp)
                new_line.extend(tmp)
                if len(new_line) > 100:
                    break
            predictions.append(new_line)

        from eval import eval_all
        print('Test', eval_all(predictions, query_ids))
        print(eval_all([predictions[j] for j in seen_split], [query_ids[j] for j in seen_split]))
        print(eval_all([predictions[j] for j in unseen_split], [query_ids[j] for j in unseen_split]))


def eval_recall(predictions, labels, subset=None):
    from eval import eval_all
    if subset is not None:
        predictions = [predictions[j] for j in subset]
        labels = [labels[j] for j in subset]
    labels = [[x] for x in labels]
    return eval_all(predictions, labels)


@torch.no_grad()
def our_encode(data_loader, model: Model, keys='doc'):
    collection = []
    code_collection = []
    for batch in tqdm(data_loader):
        batch = {k: v.cuda() for k, v in batch.items() if v is not None}
        output: QuantizeOutput = model(input_ids=batch[keys], attention_mask=batch[keys].ne(0),
                                       decoder_input_ids=batch['ids'],
                                       aux_ids=None, return_code=False,
                                       return_quantized_embedding=False, use_constraint=False)
        sentence_embeddings = output.continuous_embeds.cpu().tolist()
        code = output.probability.argmax(-1).cpu().tolist()
        code_collection.extend(code)
        collection.extend(sentence_embeddings)
    collection = np.array(collection, dtype=np.float32)
    return collection, code_collection


def norm_by_prefix(collection, prefix):
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_collection = deepcopy(collection)
    global_mean = collection.mean(axis=0)
    global_var = collection.var(axis=0)
    for p, p_code in prefix_code.items():
        p_collection = collection[p_code]
        mean_value = p_collection.mean(axis=0)
        var_value = p_collection.var(axis=0)
        var_value[var_value == 0] = 1
        scale = global_var / var_value
        scale[np.isnan(scale)] = 1
        scale = 1
        p_collection = (p_collection - mean_value + global_mean) * scale
        new_collection[p_code] = p_collection
    return new_collection


def center_pq(m, prefix):
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_m = deepcopy(m)
    for p, p_code in prefix_code.items():
        sub_m = m[p_code]
        new_m[p_code] = sub_m.mean(axis=0)
    return new_m


def norm_code_by_prefix(collection, centroids, prefix, epsilon=1):
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    attention = np.matmul(collection, centroids.T)
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    code = [None for _ in range(len(collection))]
    for p, p_code in prefix_code.items():
        p_collection = attention[p_code]
        distances = p_collection
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        distances = torch.tensor(centered_distances)
        Q = sinkhorn_raw(
            distances,
            epsilon,
            100,
            use_distrib_train=False
        )  # B-K
        codes = torch.argmax(Q, dim=-1).tolist()
        for i, c in zip(p_code, codes):
            code[i] = c
    return code


def build_index(collection, shard=True, dim=None, gpu=True):
    t = time.time()
    dim = collection.shape[1] if dim is None else dim
    cpu_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    # cpu_index = faiss.index_factory(dim, 'OPQ32,IVF1024,PQ32')
    if gpu:
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
        index = gpu_index
    else:
        index = cpu_index

    # gpu_index.train(xb)
    index.add(collection)
    print(f'build index of {len(collection)} instances, time cost ={time.time() - t}')
    return index


def do_retrieval(xq, index, k=1):
    t = time.time()
    distance, rank = index.search(xq, k)
    print(f'search {len(xq)} queries, time cost ={time.time() - t}')
    return rank, distance


def do_epoch_encode(model: Model, data, corpus, ids, tokenizer, batch_size, save_path, epoch, n_code):
    corpus_q = [['', i] for i in range(len(corpus))]
    corpus_data = BiDataset(data=corpus_q, corpus=corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=ids)
    data_loader = torch.utils.data.DataLoader(corpus_data, collate_fn=corpus_data.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)

    # collection, doc_code = our_encode(data_loader, model, 'doc')
    collection = np.zeros((100, 768))
    doc_code = [0] * len(corpus)

    print(collection.shape)
    # index = build_index(collection, gpu=False)

    q_corpus = ['' for _ in range(len(corpus))]
    corpus_data = BiDataset(data=data, corpus=q_corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=ids)
    data_loader = torch.utils.data.DataLoader(corpus_data, collate_fn=corpus_data.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    queries, query_code = our_encode(data_loader, model, 'query')

    # rank, distance = do_retrieval(queries, index, k=100)
    # rank = rank.tolist()
    rank = None
    json.dump(rank, open(f'{save_path}/{epoch}.pt.rank', 'w'))
    all_doc_code = [prefix[1:] + [current] for prefix, current in zip(ids, doc_code)]
    json.dump(all_doc_code, open(f'{save_path}/{epoch}.pt.code', 'w'))
    write_pkl(collection, f'{save_path}/{epoch}.pt.collection')

    print('Doc_code balance', balance(doc_code, ids, ncentroids=n_code))
    print('Doc_code conflict', conflict(doc_code, ids))

    # normed_collection = norm_by_prefix(collection, ids)
    nc = n_code
    # centroids, code = constrained_km(normed_collection, nc)
    code = [0] * len(corpus)
    centroids = np.zeros((nc, 768))
    print('Kmeans balance', balance(code, ids))
    print('Kmeans conflict', conflict(code, ids))
    write_pkl(centroids, f'{save_path}/{epoch}.pt.kmeans.{nc}')
    json.dump(code, open(f'{save_path}/{epoch}.pt.kmeans_code.{nc}', 'w'))

    query_ids = [x[1] for x in data]

    from eval import eval_all
    # print(eval_all(rank, query_ids))


def test_dr(config):
    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)

    dev_data = config.get('dev_data', config.get('dev_data'))
    corpus_data = config.get('corpus_data', config.get('corpus_data'))
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 128)

    data = json.load(open(dev_data))
    corpus = json.load(open(corpus_data))

    print('DR evaluation', f'{save_path}')
    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = Model(model=t5, use_constraint=False, code_length=code_length, zero_inp=False, code_number=code_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.cuda()
    model.eval()

    if prev_id is not None:
        ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        ids = [[0]] * len(corpus)

    print(len(data), len(corpus))

    for epoch in range(epochs):
        if not os.path.exists(f'{save_path}/{epoch}.pt'):
            continue
        print('#' * 20)
        print(f'DR evaluation {save_path}/{epoch}.pt')
        safe_load(model, f'{save_path}/{epoch}.pt')
        do_epoch_encode(model, data, corpus, ids, tokenizer, batch_size, save_path, epoch, n_code=code_num)


def test_case():
    batch_size = 128
    save_path = 'out/our-v12-512'
    epoch = 300
    data = json.load(open('dataset/nq320k/dev.json'))
    corpus = json.load(open('dataset/nq320k/corpus_lite.json'))

    print('DR evaluation', f'{save_path}')
    t5 = AutoModelForSeq2SeqLM.from_pretrained('models/t5-base')
    code_number = 512
    model = Model(model=t5, use_constraint=False, code_length=1, zero_inp=False, code_number=code_number)
    tokenizer = AutoTokenizer.from_pretrained('models/t5-base')
    model = model.cuda()
    model.eval()
    ids = [[0] for i, j in
           zip(json.load(open('out/our-v7-512/9.pt.code')), json.load(open('out/our-v9-512/19.pt.code')))]

    safe_load(model, f'{save_path}/{epoch}.pt')
    corpus_q = [['', i] for i in range(len(corpus))]
    corpus_data = BiDataset(data=corpus_q, corpus=corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=ids)
    data_loader = torch.utils.data.DataLoader(corpus_data, collate_fn=corpus_data.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    keys = 'doc'
    collection = []
    code_collection = []
    for batch in tqdm(data_loader):
        batch = {k: v.cuda() for k, v in batch.items() if v is not None}
        output: QuantizeOutput = model(input_ids=batch[keys], attention_mask=batch[keys].ne(0),
                                       decoder_input_ids=batch['ids'],
                                       aux_ids=None, return_code=False,
                                       return_quantized_embedding=False, use_constraint=False)
        sentence_embeddings = output.continuous_embeds.cpu().tolist()
        code = output.probability.argmax(-1).cpu().tolist()
        code_collection.extend(code)
        collection.extend(sentence_embeddings)
    collection = np.array(collection, dtype=np.float32)
    print(collection.shape)
    write_pkl(collection, f'case/l1.collection')


# centroids, code
def skl_kmeans(x, ncentroids=10, niter=300, n_init=10, mini=False, reassign=0.01):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if x.shape[0] > 1000 or mini:
        model = MiniBatchKMeans(n_clusters=ncentroids, max_iter=niter, n_init=n_init, init='k-means++', random_state=3,
                                batch_size=4096, reassignment_ratio=reassign, max_no_improvement=20, tol=1e-7,
                                verbose=1)
    else:
        model = KMeans(n_clusters=ncentroids, max_iter=niter, n_init=n_init, init='k-means++', random_state=3, tol=1e-7,
                       verbose=1)
    model.fit(x)
    return model.cluster_centers_, model.labels_.tolist()


def constrained_km(data, n_clusters=512):
    from k_means_constrained import KMeansConstrained
    size_min = min(len(data) // (n_clusters * 2), n_clusters // 4)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 2, max_iter=10, n_init=10,
                            n_jobs=10, verbose=True)
    clf.fit(data)
    return clf.cluster_centers_, clf.labels_.tolist()


def kmeans(x, ncentroids=10, niter=100):
    verbose = True
    x = np.array(x, dtype=np.float32)
    d = x.shape[1]
    model = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    model.train(x)
    D, I = model.index.search(x, 1)
    code = [i[0] for i in I.tolist()]
    return model.centroids, code


def add_last(file_in, code_num, file_out):
    corpus_ids = json.load(open(file_in))
    docid_to_doc = defaultdict(list)
    new_corpus_ids = []
    for i, item in enumerate(corpus_ids):
        docid_to_doc[str(item)].append(i)
        new_corpus_ids.append(item + [len(docid_to_doc[str(item)]) % code_num])
    json.dump(new_corpus_ids, open(file_out, 'w'))
    return new_corpus_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data03/sunweiwei-slurm/huggingface/t5-base')
    parser.add_argument('--code_num', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=3)
    parser.add_argument('--train_data', type=str, default='dataset/nq320k/train.json')
    parser.add_argument('--dev_data', type=str, default='dataset/nq320k/dev.json')
    parser.add_argument('--corpus_data', type=str, default='dataset/nq320k/corpus_lite.json')
    parser.add_argument('--save_path', type=str, default='out/model')
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


def main():
    args = parse_args()
    config = copy.deepcopy(vars(args))

    checkpoint = None

    for loop in range(args.max_length):
        config['save_path'] = args.save_path + f'-{loop + 1}-pre'
        config['code_length'] = loop + 1
        config['prev_model'] = checkpoint
        config['prev_id'] = f'{checkpoint}.code' if checkpoint is not None else None
        config['epochs'] = 1 if loop == 0 else 10
        config['loss_w'] = 1
        checkpoint = train(config)
        test_dr(config)

        config['save_path'] = args.save_path + f'-{loop + 1}'
        config['prev_model'] = checkpoint
        config['codebook_init'] = f'{checkpoint}.kmeans.{args.code_num}'
        config['epochs'] = 200
        config['loss_w'] = 2
        checkpoint = train(config)
        test_dr(config)

        test(config)

    loop = args.max_length
    config['save_path'] = args.save_path + f'-{loop}-fit'
    config['code_length'] = loop + 1
    config['prev_model'] = checkpoint
    add_last(f'{checkpoint}.code', args.code_num, f'{checkpoint}.code.last')
    config['prev_id'] = f'{checkpoint}.code.last'
    config['epochs'] = 1000
    config['loss_w'] = 3
    checkpoint = train(config)
    test_dr(config)
    test(config)
    print(checkpoint)


if __name__ == '__main__':
    main()
