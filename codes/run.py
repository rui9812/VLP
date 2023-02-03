#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
import pickle
from tqdm import *
import dgl

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='wn18rr')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-dth', '--dist_thred', default=1, type=float)
    parser.add_argument('-th', '--thred', default=0.5, type=float)
    parser.add_argument('-ga', '--gd_adv', default=1.0, type=float)
    parser.add_argument('-ma', '--mg_adv', default=1.0, type=float)
    parser.add_argument('-ct', '--cos_temp', default=1.0, type=float)
    parser.add_argument('-vw', '--vae_weight', default=0.1, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-nn', '--neighbor_num', default=4, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('-dw', '--dura_weight', default=1.5, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

args = parse_args()

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args, best_valid=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)

    if best_valid:
        save_path = args.save_path + "/best_model"
    else:
        save_path = args.save_path
    
    with open(os.path.join(save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def get_relative_distance(dirname, nentity):
    all_dist = []
    for vid in tqdm(range(nentity)):
        filename = f"./{dirname}/{vid}.pkl"
        dist_list = pickle.load(open(filename,'rb'))
        all_dist.append(dist_list)
    return all_dist

def construct_adj(triples, num_rel, num_ent):

    edge_src, edge_dst, edge_type = [], [], []

    for sub, rel, obj in triples:
        edge_src.append(sub)
        edge_type.append(rel)
        edge_dst.append(obj)

    # Adding inverse edges
    for sub, rel, obj in triples:
        edge_src.append(obj)
        edge_type.append(rel + num_rel)
        edge_dst.append(sub)

    edge_src	= torch.LongTensor(edge_src)
    edge_dst	= torch.LongTensor(edge_dst)
    edge_type	= torch.LongTensor(edge_type)

    dgl_graph = dgl.graph((edge_src, edge_dst), num_nodes=num_ent)
    dgl_graph.edata['edge_type'] = edge_type
    
    return dgl_graph

def get_relation_ht(dgl_graph):
    head, tail = dgl_graph.edges(form='uv')
    relation = dgl_graph.edata['edge_type']

    head = np.array(head)
    tail = np.array(tail)
    relation = np.array(relation)

    r_ht_dict = {}

    for edge_ids in tqdm(range(dgl_graph.num_edges())):
        h = head[edge_ids]
        t = tail[edge_ids]
        r = relation[edge_ids]
        if r not in r_ht_dict.keys():
            r_ht_dict[r] = []
        r_ht_dict[r].append((h, r, t, edge_ids))
    
    x = 0
    n = 0
    min_ht = 1000000
    for r in r_ht_dict.keys():
        n += 1
        x += len(r_ht_dict[r])
        if len(r_ht_dict[r]) <= min_ht:
            min_ht = len(r_ht_dict[r])
    
    print(x/n)
    print(min_ht)
    
    return r_ht_dict

def get_triple_relation_context_freq(dgl_graph):

    head, tail = dgl_graph.edges(form='uv')
    relation = dgl_graph.edata['edge_type']

    head = np.array(head)
    tail = np.array(tail)
    relation = np.array(relation)

    trcq_dict = {}

    for edge_ids in tqdm(range(dgl_graph.num_edges())):
        h = head[edge_ids]
        t = tail[edge_ids]
        r = relation[edge_ids]
        if t not in trcq_dict.keys():
            trcq_dict[t] = {}
        if r not in trcq_dict[t].keys():
            trcq_dict[t][r] = 0
        trcq_dict[t][r] += 1
    
    hit_num = 0
    for edge_ids in tqdm(range(dgl_graph.num_edges())):
        h = head[edge_ids]
        t = tail[edge_ids]
        r = relation[edge_ids]
        if trcq_dict[t][r] == 1:
            hit_num += 1
    
    hit_ratio = hit_num / dgl_graph.num_edges()

    print('Hit Ratio: %f'%hit_ratio)
    
    return trcq_dict
        
        
def objective():
    args.seed = 10
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    # set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Parameters: %s' % args)
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    dgl_graph = construct_adj(train_triples, nrelation, nentity)

    train_triples_with_distance = pickle.load(open(f'./data/{args.dataset}_train_list.pkl','rb'))
    valid_triples = pickle.load(open(f'./data/{args.dataset}_valid_list.pkl','rb'))
    test_triples = pickle.load(open(f'./data/{args.dataset}_test_list.pkl','rb'))
    all_relative_distance = None
    # all_relative_distance = get_relative_distance('fb237_train_e2d', nentity)
    # trcq_dict = get_triple_relation_context_freq(dgl_graph)
    r_ht_dict = get_relation_ht(dgl_graph)
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        dgl_graph=dgl_graph,
        cos_temp=args.cos_temp,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, train_triples_with_distance, dgl_graph, args.neighbor_num, args.gd_adv, r_ht_dict, all_relative_distance, nentity, nrelation, args.negative_sample_size, args.dataset, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=2,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, train_triples_with_distance, dgl_graph, args.neighbor_num, args.gd_adv, r_ht_dict, all_relative_distance, nentity, nrelation, args.negative_sample_size, args.dataset, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=2,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        best_mrr = 0
        kill_cnt = 0
        
        #Training Loop
        for step in range(init_step, args.max_steps + 1):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 2
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                kill_cnt += 1
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_relative_distance, r_ht_dict, all_true_triples, args, 'valid')
                log_metrics('Valid', step, metrics)
                if metrics['MRR'] > best_mrr:
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args, True)
                    best_mrr = metrics['MRR']
                    kill_cnt = 0

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    metrics = kge_model.test_step(kge_model, test_triples, all_relative_distance, r_ht_dict, all_true_triples, args, 'test')
                    log_metrics('Test', step, metrics)
                
            if kill_cnt >= 5:
                logging.info('Early Stop at step %d' % (step))
                break
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset with Best Model...')
        checkpoint = torch.load(os.path.join(args.save_path + "/best_model", 'checkpoint'))
        best_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        metrics = kge_model.test_step(kge_model, valid_triples, all_relative_distance, r_ht_dict, all_true_triples, args, 'valid')
        log_metrics('Valid', best_step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset with Best Model...')
        checkpoint = torch.load(os.path.join(args.save_path + "/best_model", 'checkpoint'))
        best_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        metrics = kge_model.test_step(kge_model, test_triples, all_relative_distance, r_ht_dict, all_true_triples, args, 'test')
        log_metrics('Test', best_step, metrics)
        final_result = metrics['MRR']
    
    # if args.evaluate_train:
    #     logging.info('Evaluating on Training Dataset...')
    #     metrics = kge_model.test_step(kge_model, train_triples, all_relative_distance, r_ht_dict, all_true_triples, args)
    #     log_metrics('Test', step, metrics)
    
    return final_result
        
if __name__ == '__main__':
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.save_path + "/best_model"):
        os.makedirs(args.save_path + "/best_model")
    set_logger(args)
    accuracy = objective()
