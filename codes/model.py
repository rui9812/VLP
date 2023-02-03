#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
# from torch_scatter import scatter
import dgl

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, dgl_graph, cos_temp,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation * 2
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.cos_temp = cos_temp
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.double_flag = double_relation_embedding

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation * 2, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        # self.entity_vae = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        # nn.init.uniform_(
        #     tensor=self.entity_vae, 
        #     a=-self.embedding_range.item(), 
        #     b=self.embedding_range.item()
        # )

        # self.relation_vae = nn.Parameter(torch.zeros(nrelation * 2, self.relation_dim))
        # nn.init.uniform_(
        #     tensor=self.relation_vae, 
        #     a=-self.embedding_range.item(), 
        #     b=self.embedding_range.item()
        # )

        self.relation_head = nn.Parameter(torch.zeros(nrelation * 2, self.entity_dim))
        nn.init.uniform_(
            tensor=self.relation_head, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.relation_tail = nn.Parameter(torch.zeros(nrelation * 2, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_tail, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.distance_embedding = nn.Parameter(torch.zeros(10, 1))
        nn.init.uniform_(
            tensor=self.distance_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        # self.relation_layers = torch.nn.ModuleList()

        # self.linear_1 = nn.Linear(2 * self.entity_dim + self.relation_dim, self.entity_dim, bias=False)
        # self.linear_2 = nn.Linear(self.entity_dim, self.entity_dim, bias=False)
        self.dgl_graph = dgl_graph.to('cuda')
        self.MLP_1 = nn.Linear(2*self.entity_dim, self.entity_dim)
        self.MLP_node = nn.Linear(self.entity_dim, self.entity_dim)
        self.MLP_edge = nn.Linear(self.entity_dim, self.entity_dim)
        # self.MLP_head = nn.Linear(self.entity_dim, self.entity_dim)
        self.MLP_2 = nn.Linear(self.entity_dim, self.entity_dim)
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'InterHT']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            sample, pos_dist = sample
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)

            relation_head = torch.index_select(
                self.relation_head, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)

            relation_tail = torch.index_select(
                self.relation_tail, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)

            # dist_relation = torch.index_select(
            #     self.distance_embedding, 
            #     dim=0, 
            #     index=pos_dist
            # ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        # elif mode == 'head-batch':
        #     tail_part, head_part, relative_dist = sample
        #     batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
        #     head = torch.index_select(
        #         self.entity_embedding, 
        #         dim=0, 
        #         index=head_part.view(-1)
        #     ).view(batch_size, negative_sample_size, -1)
            
        #     dist_relation = torch.index_select(
        #         self.distance_embedding, 
        #         dim=0, 
        #         index=relative_dist.contiguous().view(-1)
        #     ).view(batch_size, negative_sample_size, -1)
            
        #     relation = torch.index_select(
        #         self.relation_embedding, 
        #         dim=0, 
        #         index=tail_part[:, 1]
        #     ).unsqueeze(1)
            
        #     tail = torch.index_select(
        #         self.entity_embedding, 
        #         dim=0, 
        #         index=tail_part[:, 2]
        #     ).unsqueeze(1)
            
        elif mode == 'tail-batch' or mode == 'head-batch':
            head_part, tail_part, relative_dist = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation_head = torch.index_select(
                self.relation_head,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation_tail = torch.index_select(
                self.relation_tail,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            # dist_relation = torch.index_select(
            #     self.distance_embedding,
            #     dim=0,
            #     index=relative_dist.contiguous().view(-1)
            # ).view(batch_size, negative_sample_size, -1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'InterHT': self.InterHT
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, None, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, dist_relation, tail, mode):
        # head = F.normalize(head, dim=-1, p=2)
        # tail = F.normalize(tail, dim=-1, p=2)
        # relation_head = F.normalize(relation_head, dim=-1, p=2)
        # relation_tail = F.normalize(relation_tail, dim=-1, p=2)
        # print(relation_head.size())
        # print(relation_tail.size())
        # if mode == 'head-batch':
        #     score = head * (relation_head + 1) + (relation - tail * (relation_tail + 1))
        # else:
        #     score = (head * (relation_head + 1) + relation) - tail * (relation_tail + 1)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
    
    def InterHT(self, head, relation, dist_relation, tail, mode):
        a_head, b_head = torch.chunk(head, 2, dim=2)
        re_mid, re_tail = torch.chunk(relation, 2, dim=2)
        a_tail, b_tail = torch.chunk(tail, 2, dim=2)

        e_h = torch.ones_like(b_head)
        e_t = torch.ones_like(b_tail)

        # a_head = F.normalize(a_head, 2, -1)
        # a_tail = F.normalize(a_tail, 2, -1)
        # b_head = F.normalize(b_head, 2, -1)
        # b_tail = F.normalize(b_tail, 2, -1)
        # b_head = b_head + 1.0 * e_h
        # b_tail = b_tail + 1.0 * e_t

        score = a_head * b_tail - a_tail * b_head + re_mid
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, dist, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, dist_relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, dist_relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
    
    def get_seeds(self, dgl_graph):

        h, t = dgl_graph.edges(form='uv')
        r = dgl_graph.edata['edge_type']

        entity_embedding = F.normalize(self.entity_embedding, dim=-1, p=2)

        h_emb = entity_embedding[h]
        r_head_emb = self.relation_head[r]
        r_head_emb = r_head_emb + 1
        r_tail_emb = self.relation_tail[r]
        r_tail_emb = r_tail_emb + 1
        t_emb = entity_embedding[t]

        h_emb = h_emb * r_head_emb
        t_emb = t_emb * r_tail_emb

        # h_emb = h_emb + r_emb
        r_h_agg = scatter(h_emb, index=r, dim=0, dim_size=self.nrelation, reduce='mean')
        r_t_agg = scatter(t_emb, index=r, dim=0, dim_size=self.nrelation, reduce='mean')

        # r_h_agg = self.MLP_1(r_h_agg)
        # r_t_agg = self.MLP_2(r_t_agg)

        return r_h_agg, r_t_agg
    
    def R_VE(self, positive_sample, all_tails, pad_r_h, pad_r_t):
        entity_embedding = self.entity_embedding.detach()
        head = entity_embedding[positive_sample[:, 0]].unsqueeze(1) # (batchsize, 1, dim)
        tail = entity_embedding[all_tails] # (batchsize, neg+1, dim)
        # tail = self.entity_embedding[positive_sample[:, 2]].unsqueeze(1)
        mask = (pad_r_h != -1)

        relation = self.relation_embedding[positive_sample[:, 1]].unsqueeze(1).detach()
        # relation_weight = self.relation_head[positive_sample[:, 1]].unsqueeze(1)

        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_loop = re_head * re_relation - im_head * im_relation
        im_loop = re_head * im_relation + im_head * re_relation

        query_loop = torch.cat([re_loop, im_loop], dim = -1).squeeze(1)

        v_head = entity_embedding[pad_r_h] # (batchsize, v_num, dim)
        v_tail = entity_embedding[pad_r_t] # (batchsize, v_num, dim)   

        # head = (v_head.unsqueeze(2) - head.unsqueeze(1))
        # tail = (v_tail.unsqueeze(2) - tail.unsqueeze(1))
        head = head - v_head

        re_head = re_head * re_relation - im_head * im_relation
        im_head = re_head * im_relation + im_head * re_relation
        # re_score = re_score - re_tail
        # im_score = im_score - im_tail

        # score = torch.stack([re_score, im_score], dim = -1)
        head = torch.cat([re_head, im_head], dim = -1)

        # head = relation_weight * head

        message = self.MLP_node(v_tail) + self.MLP_edge(head)
        # print(message.size())

        # print(score.size())
        agg = message.mean(dim=1)
        # print(agg.size())
        agg = torch.tanh(self.MLP_1(torch.cat([agg, query_loop], dim=-1)))
        agg = F.normalize(agg, dim=-1, p=2)
        tail = F.normalize(tail, dim=-1, p=2)
        score = agg @ tail.transpose(1, 0)
        # score = score.norm(dim=-1).sum(dim=-1)
        # score = torch.sigmoid(6.0 - score)
        # score = score.sum(dim=1)
        # score = self.MLP_2(torch.tanh(self.MLP_1(score))).squeeze(-1)

        # score = torch.sigmoid(self.gamma-score)

        # score = 6.0 - score.sum(dim = -1)
        # return score.mean()
        # print(score.size())
        # return score * 15
        return score * self.cos_temp
    
    def C_VE(self, positive_sample, all_tails, pad_r_h, pad_r_t):
        entity_embedding = self.entity_embedding.detach()
        head = entity_embedding[positive_sample[:, 0]].unsqueeze(1) # (batchsize, 1, dim)
        tail = entity_embedding[all_tails] # (batchsize, neg+1, dim)
        # tail = self.entity_embedding[positive_sample[:, 2]].unsqueeze(1)
        mask = (pad_r_h != -1)

        relation = self.relation_embedding[positive_sample[:, 1]].unsqueeze(1).detach()
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        # relation_weight = self.relation_head[positive_sample[:, 1]].unsqueeze(1)
        
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        re_loop = re_head * re_relation - im_head * im_relation
        im_loop = re_head * im_relation + im_head * re_relation
        # re_score = re_head * re_relation - im_head * im_relation
        # im_score = re_head * im_relation + im_head * re_relation

        query_loop = torch.cat([re_loop, im_loop], dim = -1).squeeze(1)

        v_head = entity_embedding[pad_r_h] # (batchsize, v_num, dim)
        v_tail = entity_embedding[pad_r_t] # (batchsize, v_num, dim)   

        # head = (v_head.unsqueeze(2) - head.unsqueeze(1))
        # tail = (v_tail.unsqueeze(2) - tail.unsqueeze(1))
        head = head - v_head

        re_head = re_head * re_relation - im_head * im_relation
        im_head = re_head * im_relation + im_head * re_relation
        # re_score = re_score - re_tail
        # im_score = im_score - im_tail

        # score = torch.stack([re_score, im_score], dim = -1)
        head = torch.cat([re_head, im_head], dim = -1)

        # head = relation_weight * head

        message = self.MLP_node(v_tail) + self.MLP_edge(head)
        # print(message.size())

        # print(score.size())
        agg = message.mean(dim=1)
        # print(agg.size())
        agg = torch.tanh(self.MLP_1(torch.cat([agg, query_loop], dim=-1)))
        agg = F.normalize(agg, dim=-1, p=2)
        tail = F.normalize(tail, dim=-1, p=2)
        score = agg @ tail.transpose(1, 0)

        return score * self.cos_temp

    def D_VE(self, positive_sample, all_tails, pad_r_h, pad_r_t):
        entity_embedding = self.entity_embedding.detach()
        head = entity_embedding[positive_sample[:, 0]].unsqueeze(1) # (batchsize, 1, dim)
        tail = entity_embedding[all_tails] # (batchsize, neg+1, dim)
        # tail = self.entity_embedding[positive_sample[:, 2]].unsqueeze(1)
        mask = (pad_r_h != -1)

        relation = self.relation_embedding[positive_sample[:, 1]].unsqueeze(1).detach()
        # re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        # # relation_weight = self.relation_head[positive_sample[:, 1]].unsqueeze(1)
        
        # re_head, im_head = torch.chunk(head, 2, dim=-1)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # re_loop = re_head * re_relation - im_head * im_relation
        # im_loop = re_head * im_relation + im_head * re_relation
        # re_score = re_head * re_relation - im_head * im_relation
        # im_score = re_head * im_relation + im_head * re_relation

        # query_loop = torch.cat([re_loop, im_loop], dim = -1).squeeze(1)
        query_loop = (head * relation).squeeze(1)

        v_head = entity_embedding[pad_r_h] # (batchsize, v_num, dim)
        v_tail = entity_embedding[pad_r_t] # (batchsize, v_num, dim)   

        # head = (v_head.unsqueeze(2) - head.unsqueeze(1))
        # tail = (v_tail.unsqueeze(2) - tail.unsqueeze(1))
        head = head - v_head

        # re_head = re_head * re_relation - im_head * im_relation
        # im_head = re_head * im_relation + im_head * re_relation
        # # re_score = re_score - re_tail
        # # im_score = im_score - im_tail

        # # score = torch.stack([re_score, im_score], dim = -1)
        # head = torch.cat([re_head, im_head], dim = -1)
        head = head * relation

        # head = relation_weight * head

        message = self.MLP_node(v_tail) + self.MLP_edge(head)
        # print(message.size())

        # print(score.size())
        agg = message.mean(dim=1)
        # print(agg.size())
        agg = torch.tanh(self.MLP_1(torch.cat([agg, query_loop], dim=-1)))
        agg = F.normalize(agg, dim=-1, p=2)
        tail = F.normalize(tail, dim=-1, p=2)
        score = agg @ tail.transpose(1, 0)

        return score * 15
    
    def VE(self, positive_sample, all_tails, pad_r_h, pad_r_t):
        
        head = self.entity_embedding[positive_sample[:, 0]].unsqueeze(1) # (batchsize, dim)
        tail = self.entity_embedding[positive_sample[:, 2]].unsqueeze(1) # (batchsize, neg+1, dim)
        mask = (pad_r_h != -1)

        # head = F.normalize(head, dim=-1, p=2).unsqueeze(1)
        # tail = F.normalize(tail, dim=-1, p=2)

        # r_head_emb = self.relation_head[positive_sample[:, 1]].unsqueeze(1)
        # # r_head_emb = r_head_emb + 1
        # r_tail_emb = self.relation_tail[positive_sample[:, 1]].unsqueeze(1)
        # r_tail_emb = r_tail_emb + 1

        v_head = self.entity_embedding[pad_r_h] # (batchsize, v_num, dim)
        v_tail = self.entity_embedding[pad_r_t] # (batchsize, v_num, dim)

        # head = head * (r_head_emb + 1)
        # v_head = v_head * (r_head_emb + 1)

        # tail = tail * (r_tail_emb + 1)
        # v_tail = v_tail * (r_tail_emb + 1)
        
        # head_bias = head - v_head # (batchsize, v_num, dim)

        head = v_head - head
        tail = v_tail - tail
        
        # v_tail = v_tail + head_bias
        
        score = head - tail

        return score.norm(dim=-1).mean()

        # head = F.normalize(head, dim=-1, p=2)
        # tail = F.normalize(tail, dim=-1, p=2)

        # # tail = v_tail @ tail.permute(0, 2, 1)
        # tail = (head * tail).sum(-1)

        # # tail = (v_tail.unsqueeze(2) - tail.unsqueeze(1)).norm(dim=-1, p=1)

        # tail = tail * mask

        # tail = tail.sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # # score = 6.0 - tail
        # score= tail * 6.0

        # return score
        

    def VertE(self, head, r_head, tail, r_tail, r_h_agg, r_t_agg):
        # vert_head (batch, vert_num, dim)
        # vert_tail (batch, vert_num, dim)
        # head (batch, 1, dim)
        # tail (batch, 1, dim)
        # relation (batch, 1, dim)
        # r_h_agg = r_h_agg.unsqueeze(1)
        # r_t_agg = r_t_agg.unsqueeze(1)
        head = F.normalize(head, dim=-1, p=2)
        tail = F.normalize(tail, dim=-1, p=2)
        head = head * (r_head)

        tail = tail * (r_tail)

        head = head - r_h_agg # (batch, 1, dim)
        # print(head.size())
        # tail = r_t_agg - tail
        head = r_t_agg + head
        # print(tail.size())
        tail = head - tail
        # tail = torch.cat([head.repeat(1, tail.size(1), 1), tail], dim=-1)
        # print(tail.size())

        score = 6.0 - (torch.norm(tail, p=1, dim=-1))
        # head = F.normalize(head, dim=-1, p=2)
        # tail = F.normalize(tail, dim=-1, p=2)
        # score = (head * tail).sum(-1)
        # score = self.MLP_2(torch.relu(self.MLP_1(tail))).squeeze(-1)
        
        return score


    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    def N3(self, positive_sample, weight):
        norm = 0
        
        h = self.entity_embedding[positive_sample[:, 0]]
        r = self.relation_embedding[positive_sample[:, 1]]
        t = self.entity_embedding[positive_sample[:, 2]]
        
        if self.double_flag:
            h = self.entity_embedding[positive_sample[:, 0]]
            re_h, im_h = torch.chunk(h, 2, dim=-1)
            h = torch.sqrt(re_h ** 2 + im_h ** 2)
            r = self.relation_embedding[positive_sample[:, 1]]
            re_r, im_r = torch.chunk(r, 2, dim=-1)
            r = torch.sqrt(re_r ** 2 + im_r ** 2)
            t = self.entity_embedding[positive_sample[:, 2]]
            re_t, im_t = torch.chunk(t, 2, dim=-1)
            t = torch.sqrt(re_t ** 2 + im_t ** 2)
        
        factors = [(h, r, t)]
        for factor in factors:
            for f in factor:
                norm += weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        return norm
    
    def cal_posterior_weight(self, positive_score, negative_score, adv, thred, margin_adv):
        # positive_score (batchsize, 1)
        # negative_score (batchsize, num_negs)
        
        # negative_weight = torch.exp(negative_score * adv)
        # positive_weight = torch.exp((positive_score+thred) * adv)
        margin_thred = thred
        positive_weight = positive_score + margin_thred

        mask = (negative_score >= (positive_score + margin_thred))
        # print(positive_score)
        # print(negative_score)
        # print(mask)
        margin = (negative_score - positive_score - margin_thred).clamp(min=0)
        # margin_weight = torch.exp(- margin * margin_adv)
        # margin_weight = margin * margin_adv
        margin_weight = positive_weight * adv - margin * margin_adv

        # margin_weight = margin_weight - 1
        # margin_weight = (positive_weight + margin_weight).clamp(min=0)
        negative_weight = torch.where(mask, margin_weight, negative_score*adv)
        # negative_weight = negative_score
        negative_weight = F.softmax(negative_weight, dim=1)
        # sum_weight = negative_weight.sum(-1, keepdim=True)
        # negative_weight = negative_weight / (sum_weight).clamp(min=1e-9)
        # margin = (positive_score + thred - negative_score).abs()
        # negative_weight = - margin * margin_adv
        # negative_weight = F.softmax(negative_weight, dim=1)

        return negative_weight
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, pad_r_ht_tensor, subsampling_weight, eids_to_exclude, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            pad_r_ht_tensor = pad_r_ht_tensor.cuda()
            # hr_t = hr_t.cuda()
            subsampling_weight = subsampling_weight.cuda()
            eids_to_exclude = eids_to_exclude.cuda()
        
        # print(pad_r_ht_tensor)
        # print(pad_r_ht_tensor.size())
        # print(positive_sample)
        pad_r_h = pad_r_ht_tensor[:, :, 0]
        pad_r_t = pad_r_ht_tensor[:, :, 2]
        # in_batch_mask = (positive_sample[:, 1].unsqueeze(1) == positive_sample[:, 1].unsqueeze(0))
        # g = dgl.remove_edges(model.dgl_graph, eids_to_exclude)
        # r_h_agg, r_t_agg = model.get_seeds(model.dgl_graph)

        # all_distance = torch.clamp(all_distance, max=9)
        # negative_distance = all_distance[:, 1:]
        # vert_head = model.entity_embedding[positive_sample[:, 0]]
        # vert_tail = model.entity_embedding[positive_sample[:, 2]]
        # # print(positive_sample)
        # # print(pad_r_ht_tensor)

        # all_tails = torch.cat([positive_sample, negative_sample], dim=-1)[:, 2:]
        all_tails = torch.arange(model.entity_embedding.size(0)).cuda()
        # score = model.VertE(model.entity_embedding[positive_sample[:, 0]].unsqueeze(1), model.relation_head[positive_sample[:, 1]].unsqueeze(1), model.entity_embedding[all_tails], model.relation_tail[positive_sample[:, 1]].unsqueeze(1), r_h_agg[positive_sample[:, 1]].unsqueeze(1), r_t_agg[positive_sample[:, 1]].unsqueeze(1))
        # score = model.D_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
        if args.model == 'RotatE':
            score = model.R_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
        elif args.model == 'ComplEx':
            score = model.C_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
        elif args.model == 'DistMult':
            score = model.D_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
        # print(score.size())
        # reg = reg * 0.1
        # score = score.view(-1, score.size(-1))
        # print(score.size())
        # soft_score = F.softmax(score, dim=-1)
        # print(soft_score)
        v_positive_score = score[:, 0]
        v_negative_score = score[:, 1:]
        negative_score = model((positive_sample, negative_sample, None), mode=mode)
        # negative_score = negative_score + v_negative_score

        # scores = torch.cat([positive_score, negative_score], dim=1)
        # print(score.size())
        # label = torch.zeros(score.size(0), device=score.device).long()
        # score = torch.masked_fill(score, hr_t.bool(), value=-1e9)
        label = positive_sample[:, 2]
        loss_v = F.cross_entropy(score, target=label)


        positive_score = model((positive_sample, None))

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_weight = model.cal_posterior_weight(positive_score.detach(), negative_score.detach(), args.adversarial_temperature, args.thred, args.mg_adv)
            negative_score = (negative_weight * F.logsigmoid(-negative_score)).sum(dim = 1)
            # v_negative_score = (F.softmax(v_negative_score * args.adversarial_temperature, dim = 1).detach() 
            #                   * F.logsigmoid(-v_negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        # positive_distance = all_distance[:, 0]
        # positive_score = model((positive_sample, None))
        # # positive_score = positive_score + v_positive_score.unsqueeze(1)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        # v_positive_score = F.logsigmoid(score).squeeze(dim = -1)
        # v_positive_score = F.logsigmoid(v_positive_score)
        # v_positive_sample_loss = - v_positive_score.mean()

        # v_positive_sample_loss = - v_positive_score.mean()
        # v_negative_sample_loss = - v_negative_score.mean()
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
            # v_positive_sample_loss = - v_positive_score.mean()
            # v_negative_sample_loss = - v_negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
            # v_positive_sample_loss = - (subsampling_weight * v_positive_score).sum()/subsampling_weight.sum()
            # v_negative_sample_loss = - (subsampling_weight * v_negative_score).sum()/subsampling_weight.sum()

        loss_h = (positive_sample_loss + negative_sample_loss)/2
        # loss_v = (v_positive_sample_loss + v_negative_sample_loss)/2

        loss = loss_h + args.vae_weight * loss_v

        # reg = 0.1 * (model.entity_embedding.norm(dim=-1, p=2).mean())
        
        if args.regularization != 0.0:
            # Use N3 regularization for ComplEx and DistMult
            regularization = model.N3(positive_sample, args.regularization)
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss_v': loss_v.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_relative_distance, r_ht_dict, all_true_triples, args, torv):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_relative_distance,
                    args.neighbor_num, 
                    r_ht_dict,
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    args.dataset,
                    torv,
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_relative_distance,
                    args.neighbor_num, 
                    r_ht_dict,
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    args.dataset,
                    torv,
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, pad_r_ht_tensor, filter_bias, mode, distance, all_distance in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            pad_r_ht_tensor = pad_r_ht_tensor.cuda()
                            filter_bias = filter_bias.cuda()
                            all_distance = all_distance.cuda()
                            distance_cuda = distance.cuda()

                        batch_size = positive_sample.size(0)
                        all_distance = torch.clamp(all_distance, max=9)

                        pad_r_h = pad_r_ht_tensor[:, :, 0]
                        pad_r_t = pad_r_ht_tensor[:, :, 2]

                        # print(mode)
                        # print(pad_r_ht_tensor)
                        # print(positive_sample)
                        # print(pad_r_h)
                        # print(pad_r_t)

                        # r_h_agg, r_t_agg = model.get_seeds(model.dgl_graph)
                        # all_tails = torch.cat([positive_sample, negative_sample], dim=-1)[:, 2:]
                        all_tails = torch.arange(model.entity_embedding.size(0)).cuda()
                        if args.model == 'RotatE':
                            v_score = model.R_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
                        elif args.model == 'ComplEx':
                            v_score = model.C_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
                        elif args.model == 'DistMult':
                            v_score = model.D_VE(positive_sample, all_tails, pad_r_h, pad_r_t)
                        v_score = F.softmax(v_score, dim=-1)

                        score = model((positive_sample, negative_sample, all_distance), mode)
                        score = torch.sigmoid(score)
                        # score = v_score
                        weight = (distance_cuda.unsqueeze(1) <= args.dist_thred).long()
                        score = weight * score + (1.0 - weight) * v_score
                        
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 2]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            # logs.append({
                            #     'MRR': 1.0/ranking,
                            #     'MR': float(ranking),
                            #     'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            #     'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            #     'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            # })

                            result = {
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            }
                            logs.append(result)

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
