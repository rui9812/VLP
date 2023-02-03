#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import pickle
import dgl

from torch.utils.data import Dataset
import torch.nn.functional as F
from graph_tool.all import Graph, shortest_distance
from tqdm import tqdm

# all_distance_np = torch.from_numpy(np.load(f"./wn18rr_distance.npy"))

class TrainDataset(Dataset):
    def __init__(self, triples, triples_with_distance, dgl_graph, neighbor_num, gd_adv, r_ht_dict, all_relative_distance, nentity, nrelation, negative_sample_size, dataset, mode):
        self.len = len(triples_with_distance)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        # self.hr_t_dict = self.get_hr_t()
        self.r_ht_dict = r_ht_dict
        self.triples_with_distance = triples_with_distance
        self.all_relative_distance = all_relative_distance
        self.dgl_graph = dgl_graph
        self.hr_t_dict = self.get_hr_t()
        self.gt_graph = self.dgl_graph_to_gt_graph(dgl_graph)
        self.path_name =f'./{dataset}_distance.npy'
        if not os.path.exists(self.path_name):
            print('Processing Distance...')
            # os.mkdir(self.h_path_name)
            self.all_distance_np = torch.from_numpy(self.get_dist_dict())
        else:
            self.all_distance_np = torch.from_numpy(np.load(self.path_name))
        self.ht_num = neighbor_num
        self.gd_adv = gd_adv
        self.ht_name = f'./{dataset}-vr-100.pt'
        if not os.path.exists(self.ht_name):
            print('Processing VAG...')
            # os.mkdir(self.h_path_name)
            self.r_ht = self.get_r_ht()
        else:
            self.r_ht = torch.load(self.ht_name)
        print(self.r_ht.size())
    
    def get_hr_t(self):
        hr_t_dict = {}
        head, tail = self.dgl_graph.edges(form='uv')
        relation = self.dgl_graph.edata['edge_type']
        head = np.array(head)
        tail = np.array(tail)
        relation = np.array(relation)
        for edge_id in tqdm(range(self.dgl_graph.num_edges())):
            h = head[edge_id]
            t = tail[edge_id]
            r = relation[edge_id]
            if (h, r) not in hr_t_dict.keys():
                hr_t_dict[(h, r)] = []
            hr_t_dict[(h, r)].append(t)
        return hr_t_dict
    
    def get_r_ht(self):
        r_ht = []
        head, tail = self.dgl_graph.edges(form='uv')
        relation = self.dgl_graph.edata['edge_type']

        # head = head.cuda()
        # tail = tail.cuda()
        # relation = relation.cuda()
        head = np.array(head)
        tail = np.array(tail)
        relation = np.array(relation)
        all_distance_cuda = self.all_distance_np.cuda()
        print(all_distance_cuda.size())
        for edge_id in tqdm(range(self.dgl_graph.num_edges())):
            h = head[edge_id]
            t = tail[edge_id]
            r = relation[edge_id]
            dist_list = all_distance_cuda[h]
            r_ht_list = self.r_ht_dict[r]
            r_ht_tensor = torch.LongTensor(r_ht_list).cuda()
            # print(r_ht_tensor.size())
            r_ht_tensor = r_ht_tensor[r_ht_tensor[:, 3] != edge_id]
            # print(r_ht_tensor.size())
            r_h = r_ht_tensor[:, 0]
            if len(r_ht_list) <= 20:
                pad_r_ht_tensor = F.pad(input=r_ht_tensor, pad=(0, 0, 0, 20-r_ht_tensor.size(0)), mode='constant', value=-1)
            else:    
                topk_dist, topk_indices = torch.topk(dist_list[r_h], k=20, dim=-1, largest=False)
                # print(topk_dist)
                pad_r_ht_tensor = r_ht_tensor[topk_indices, :]
            r_ht.append(pad_r_ht_tensor)
        r_ht = torch.stack(r_ht, dim=0).cpu()
        # print(r_ht.size())
        # print(r_ht)
        torch.save(r_ht, self.ht_name)
        return r_ht

    
    def dgl_graph_to_gt_graph(self, dgl_graph, directed=True):
        row, col = dgl_graph.edges()
        edges = torch.cat([row.view(-1,1), col.view(-1,1)],dim=-1)
        gt_g = Graph()
        gt_g.add_vertex(int(dgl_graph.num_nodes()))
        gt_g.add_edge_list(edges.numpy())
        gt_g.set_directed(directed)
        return gt_g
    
    def get_dist_dict(self):
        dist = []
        for node_id in tqdm(range(self.dgl_graph.num_nodes())):
            source_dist = shortest_distance(self.gt_graph, source = node_id, max_dist=8)
            source_dist = source_dist.a
            source_dist[source_dist>8] = 9
            dist.append(source_dist)
        distance_array = np.array(dist, dtype=np.int8)
        print(distance_array.shape)
        np.save(self.path_name, distance_array)
        return distance_array
    
    def get_label(self, label, pos_t):
        y = np.zeros([self.nentity], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        y[pos_t] = 0.0
        return torch.LongTensor(y)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # positive_sample = self.triples[idx]

        # head, relation, tail = positive_sample
        head, relation, tail = self.triples_with_distance[idx]['triple']
        pos_dist = torch.LongTensor([self.triples_with_distance[idx]['distance']])

        # print(pad_r_ht_tensor.size())

        # positive_sample = torch.LongTensor([head, relation, tail])

        if self.mode == 'tail-batch':
            positive_sample = torch.LongTensor([head, relation, tail])
            dist_list = self.all_distance_np[head]
            # pos_seed = torch.LongTensor([tail])
            # dist_list[head] = 1
            eid = idx
            inv_eid = idx + self.len
            # r_ht_list = self.r_ht_dict[relation]
            # hr_t = self.hr_t_dict[(head, relation)]
            # hr_t = self.get_label(hr_t, tail)
        else:
            positive_sample = torch.LongTensor([tail, relation + self.nrelation, head])
            dist_list = self.all_distance_np[tail]
            # pos_seed = torch.LongTensor([head])
            # dist_list[tail] = 1
            eid = idx + self.len
            inv_eid = idx
            # r_ht_list = self.r_ht_dict[relation + self.nrelation]
            # hr_t = self.hr_t_dict[(tail, relation + self.nrelation)]
            # hr_t = self.get_label(hr_t, head)
        
        # print(hr_t)
        # hr_t = self.get_label(hr_t, )
        # print(hr_t.size())
        # r_ht_tensor = torch.LongTensor(r_ht_list)
        # r_ht_tensor = r_ht_tensor[r_ht_tensor[:, 3] != eid]
        # r_h = r_ht_tensor[:, 0]
        # if len(r_ht_list) <= 10:
        #     pad_r_ht_tensor = F.pad(input=r_ht_tensor, pad=(0, 0, 0, 10-r_ht_tensor.size(0)), mode='constant', value=-1)
        # else:    
        #     topk_dist, topk_indices = torch.topk(dist_list[r_h], k=10, dim=-1, largest=False)
        #     pad_r_ht_tensor = r_ht_tensor[topk_indices, :]
        pad_r_ht_tensor = self.r_ht[eid][:self.ht_num]
        # print(pad_r_ht_tensor)
        # print(positive_sample)
        # print(dist_list.size())
        # print(dist_list[pad_r_ht_tensor[:, 0]])
            # print(topk_dist)

        # r_ht_list = self.r_ht_dict[relation]
        # r_ht_tensor = torch.LongTensor(r_ht_list)
        # r_ht_tensor = r_ht_tensor[r_ht_tensor[:, 3] != eid]

        # rand_index = np.random.randint(len(r_ht_list), size=128)
        # pad_r_ht_tensor = r_ht_tensor[rand_index]
        # r_ht_tensor = torch.from_numpy(r_ht_np[rand_index])
        # print(len(r_ht_list))
        # r_ht_tensor = torch.LongTensor(r_ht_list)
        # print(r_ht_tensor.size())

        # if len(r_ht_list) <= 50:
        #     pad_r_ht_tensor = F.pad(input=r_ht_tensor, pad=(0, 0, 0, 50-r_ht_tensor.size(0)), mode='constant', value=-1)
            # print(pad_r_ht_tensor.size())
        # else:
        # rand_index = torch.randint(r_ht_tensor.size(0), size=(5,))
        # pad_r_ht_tensor = r_ht_tensor[rand_index]
        
        # print(pos_seed)
        # dist_list[dist_list>=7] = 1e12
        dist_list[positive_sample[0]] = 1
        dist_list[dist_list>=8] = 8
        # p = np.array(torch.nn.functional.softmax(- dist_list * 2.0, dim=0))
        p = np.array(torch.nn.functional.softmax(- dist_list * self.gd_adv, dim=0))
        # p[-1] = 1 - p[0:-1].sum()

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            # negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            negative_sample = np.random.choice(self.nentity, size=self.negative_sample_size*2, replace=False, p=p)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)
        
        # if self.mode == 'tail-batch':
        #     positive_sample = torch.LongTensor([head, relation, tail])
        # else:
        #     positive_sample = torch.LongTensor([tail, relation + self.nrelation, head])
        # positive_sample = torch.LongTensor([head, relation, tail])
        
        # dist_list = torch.LongTensor(self.get_relative_distance(positive_sample[0]))
        # if self.mode == 'tail-batch':
        #     dist_list = torch.LongTensor(self.all_relative_distance[positive_sample[0]])
        # else:
        #     dist_list = torch.LongTensor(self.all_relative_distance[positive_sample[2]])
        
        # neg_dist = dist_list[negative_sample]

        # all_dist = torch.cat([pos_dist, neg_dist], dim=0)
        # all_seed = torch.cat([pos_seed, negative_sample], dim=0)
        # subgraph = self.sample_subgraph(all_seed, all_dist, dist_list)
        # print(subgraph)

        # pos_dist[pos_dist > 10] = 10
        # neg_dist[neg_dist > 10] = 10
        eids_to_exclude = torch.LongTensor([eid])

        return positive_sample, negative_sample, pad_r_ht_tensor, subsampling_weight, eids_to_exclude, self.mode
    
    def sample_subgraph(self, seed_nodes, seed_dist, dist_list):
        # induced_nodes = {0: seed_nodes.view(-1)}
        cur_nodes = seed_nodes
        cur_dist = seed_dist
        dist_init = 3
        dist_bound = 0
        edges_to_exclude = []
        edge_ids_list = []
        for i in range(2):
            if i == 0:
                pre_nodes = cur_nodes[cur_dist >= dist_init]
            elif i == 1:
                pre_nodes = cur_nodes[cur_dist < (pre_dist + dist_bound)]
            else:
                pre_nodes = cur_nodes[cur_dist < pre_dist]
            sub_g = dgl.sampling.sample_neighbors(self.dgl_graph, pre_nodes, -1, exclude_edges=edges_to_exclude)
            cur_nodes, pre_nodes = sub_g.edges()
            sampled_edge_ids = sub_g.edata[dgl.EID]
            edge_ids_list.append(sampled_edge_ids)
            
            cur_dist = dist_list[cur_nodes]
            pre_dist = dist_list[pre_nodes]
            out_sub_g = dgl.out_subgraph(self.dgl_graph, pre_nodes.unique(), relabel_nodes=False)
            edges_to_exclude = out_sub_g.edata[dgl.EID]
            # induced_nodes[i + 1] = cur_nodes.unique()
        eids = torch.cat(edge_ids_list, dim=0).unique()
        subgraph = dgl.edge_subgraph(self.dgl_graph, eids, relabel_nodes=False)
        return subgraph
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        pad_r_ht_tensor = torch.stack([_[2] for _ in data], dim=0)
        # hr_t = torch.stack([_[3] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        eids_to_exclude = torch.cat([_[4] for _ in data], dim=0).unique()
        mode = data[0][5]
        return positive_sample, negative_sample, pad_r_ht_tensor, subsample_weight, eids_to_exclude, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_relative_distance, neighbor_num, r_ht_dict, all_true_triples, nentity, nrelation, dataset, torv, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.r_ht_dict = r_ht_dict
        self.mode = mode
        self.all_relative_distance = all_relative_distance
        self.ht_num = neighbor_num
        self.path_name =f'./{dataset}_distance.npy'
        self.all_distance_np = torch.from_numpy(np.load(self.path_name))
        if torv == 'test':
            self.ht_name_head_batch = f'./{dataset}-vr-test-head_batch.pt'
            self.ht_name_tail_batch = f'./{dataset}-vr-test-tail_batch.pt'
        else:
            self.ht_name_head_batch = f'./{dataset}-vr-valid-head_batch.pt'
            self.ht_name_tail_batch = f'./{dataset}-vr-valid-tail_batch.pt'
        if not os.path.exists(self.ht_name_head_batch):
            print('Processing VAG...')
            # os.mkdir(self.h_path_name)
            self.r_ht_tail_batch, self.r_ht_head_batch = self.get_r_ht_test()
        else:
            self.r_ht_tail_batch = torch.load(self.ht_name_tail_batch)
            self.r_ht_head_batch = torch.load(self.ht_name_head_batch)
        # print(self.r_ht.size())
    
    def get_r_ht_test(self):
        r_ht_tail_batch = []
        r_ht_head_batch = []
        # head, tail = self.dgl_graph.edges(form='uv')
        # relation = self.dgl_graph.edata['edge_type']
        
        # head = np.array(head)
        # tail = np.array(tail)
        # relation = np.array(relation)
        all_distance_cuda = self.all_distance_np.cuda()
        print(all_distance_cuda.size())
        for test_id in tqdm(range(self.len)):
            # h = head[edge_id]
            # t = tail[edge_id]
            # r = relation[edge_id]
            h, r, t = self.triples[test_id]['triple']

            # 'tail-batch mode'
            dist_list = all_distance_cuda[h]
            r_ht_list = self.r_ht_dict[r]
            r_ht_tensor = torch.LongTensor(r_ht_list).cuda()
            # print(r_ht_tensor.size())
            # r_ht_tensor = r_ht_tensor[r_ht_tensor[:, 3] != edge_id]
            # print(r_ht_tensor.size())
            r_h = r_ht_tensor[:, 0]
            if len(r_ht_list) <= 20:
                pad_r_ht_tensor = F.pad(input=r_ht_tensor, pad=(0, 0, 0, 20-r_ht_tensor.size(0)), mode='constant', value=-1)
            else:    
                topk_dist, topk_indices = torch.topk(dist_list[r_h], k=20, dim=-1, largest=False)
                # print(topk_dist)
                pad_r_ht_tensor = r_ht_tensor[topk_indices, :]
            r_ht_tail_batch.append(pad_r_ht_tensor)

            # head-batch mode
            dist_list = all_distance_cuda[t]
            r_ht_list = self.r_ht_dict[r+self.nrelation]
            r_ht_tensor = torch.LongTensor(r_ht_list).cuda()
            r_h = r_ht_tensor[:, 0]
            if len(r_ht_list) <= 20:
                pad_r_ht_tensor = F.pad(input=r_ht_tensor, pad=(0, 0, 0, 20-r_ht_tensor.size(0)), mode='constant', value=-1)
            else:    
                topk_dist, topk_indices = torch.topk(dist_list[r_h], k=20, dim=-1, largest=False)
                # print(topk_dist)
                pad_r_ht_tensor = r_ht_tensor[topk_indices, :]
            r_ht_head_batch.append(pad_r_ht_tensor)
        
        r_ht_tail_batch = torch.stack(r_ht_tail_batch, dim=0).cpu()
        r_ht_head_batch = torch.stack(r_ht_head_batch, dim=0).cpu()
        # print(r_ht.size())
        # print(r_ht)
        torch.save(r_ht_tail_batch, self.ht_name_tail_batch)
        torch.save(r_ht_head_batch, self.ht_name_head_batch)
        return r_ht_tail_batch, r_ht_head_batch

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]['triple']
        distance = self.triples[idx]['distance']
        # if distance <= 2:
            # distance = 2
        if distance >= 6:
            distance = 6
        # if distance > 100:
            # distance = 7
        distance = torch.LongTensor([distance])

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        # positive_sample = torch.LongTensor((head, relation, tail))
        # if self.mode == 'tail-batch':
        #     positive_sample = torch.LongTensor([head, relation, tail])
        # else:
        #     positive_sample = torch.LongTensor([tail, relation + self.nrelation, head])
        if self.mode == 'tail-batch':
            positive_sample = torch.LongTensor([head, relation, tail])
            dist_list = self.all_distance_np[head]
            pad_r_ht_tensor = self.r_ht_tail_batch[idx][:self.ht_num]
        else:
            positive_sample = torch.LongTensor([tail, relation + self.nrelation, head])
            dist_list = self.all_distance_np[tail]
            pad_r_ht_tensor = self.r_ht_head_batch[idx][:self.ht_num]
        
        all_dist = dist_list[negative_sample]

        # all_dist[all_dist < 2] = 0
        # all_dist[all_dist > 10] = 10
            
        return positive_sample, negative_sample, pad_r_ht_tensor, filter_bias, self.mode, distance, all_dist
    
    def get_relative_distance(self, vid):
        filename = f"./{self.path_dist_file}/{vid}.pkl"
        dist_list = pickle.load(open(filename,'rb'))
        return dist_list
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        pad_r_ht_tensor = torch.stack([_[2] for _ in data], dim=0)
        filter_bias = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        distance = torch.cat([_[5] for _ in data], dim=0)
        all_distance = torch.stack([_[6] for _ in data], dim=0)
        return positive_sample, negative_sample, pad_r_ht_tensor, filter_bias, mode, distance, all_distance
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data