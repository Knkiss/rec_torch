# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：dataLoader.py
@Author  ：Knkiss
@Date    ：2023/2/14 10:13 
"""
import collections
import os
import random
from os.path import join
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, coo_matrix

import scipy.sparse as sp

from train import utils
import world


class UIDataset(Dataset):
    def __init__(self, path=None):
        if path is None:
            path = join(world.PATH_DATA, world.dataset)
        print(f'loading [{path}]')
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    if len(line) == 1:
                        continue
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    if line[1]:
                        items = [int(i) for i in line[1:]]
                        uid = int(line[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        if os.path.exists(valid_file):
            with open(valid_file) as f:
                for line in f.readlines():
                    if len(line) > 0:
                        line = line.strip('\n').split(' ')
                        if line[1]:
                            items = [int(i) for i in line[1:]]
                            uid = int(line[0])
                            validUniqueUsers.append(uid)
                            validUser.extend([uid] * len(items))
                            validItem.extend(items)
                            self.m_item = max(self.m_item, max(items))
                            self.n_user = max(self.n_user, uid)
                            self.validDataSize += len(items)
            self.validUniqueUsers = np.array(validUniqueUsers)
            self.validUser = np.array(validUser)
            self.validItem = np.array(validItem)

        self.m_item += 1
        self.n_user += 1
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except Exception:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            self.Graph = utils.convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg


class KGDataset(Dataset):
    def __init__(self, kg_path=None):
        if kg_path is None:
            kg_path = join(world.PATH_DATA, world.dataset, "kg.txt")
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data()
        self.item_net_path = join(world.PATH_DATA, world.dataset)
        self.length = len(self.kg_dict)

    @property
    def entity_count(self):
        return self.kg_data['t'].max() + 2

    @property
    def relation_count(self):
        return self.kg_data['r'].max() + 2

    def get_kg_graph(self):
        head = self.kg_data['h'].values
        tail = self.kg_data['t'].values + head.max()
        value = self.kg_data['r'].values
        kg_graph = coo_matrix((value, (head, tail)), shape=(tail.max() + 1, tail.max() + 1))
        return kg_graph

    def get_kg_dict(self, item_num):
        entity_num = world.hyper_KGDataset_entity_num_per_item
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x: x[1], rts))
                relations = list(map(lambda x: x[0], rts))
                if len(tails) > entity_num:
                    i2es[item] = torch.IntTensor(tails).to(world.device)[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(world.device)[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count] * (entity_num - len(tails)))
                    relations.extend([self.relation_count] * (entity_num - len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(world.device)
                    i2rs[item] = torch.IntTensor(relations).to(world.device)
            else:
                i2es[item] = torch.IntTensor([self.entity_count] * entity_num).to(world.device)
                i2rs[item] = torch.IntTensor([self.relation_count] * entity_num).to(world.device)
        return i2es, i2rs

    def generate_kg_data(self):
        kg_dict = collections.defaultdict(list)

        # DIFF 性能未知 部分relation对性能有一定提升
        relation_list = 'All'
        for row in self.kg_data.iterrows():
            h, r, t = row[1]
            if relation_list == 'All' or r in relation_list:
                kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def get_kg_dict_random(self, item_num):
        kg_dict = collections.defaultdict(list)
        a = self.kg_data.sample(frac=0.8)
        for row in a.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))

        entity_num = world.hyper_KGDataset_entity_num_per_item
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x: x[1], rts))
                relations = list(map(lambda x: x[0], rts))
                if len(tails) > entity_num:
                    i2es[item] = torch.IntTensor(tails).to(world.device)[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(world.device)[:entity_num]
                else:
                    tails.extend([self.entity_count] * (entity_num - len(tails)))
                    relations.extend([self.relation_count] * (entity_num - len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(world.device)
                    i2rs[item] = torch.IntTensor(relations).to(world.device)
            else:
                i2es[item] = torch.IntTensor([self.entity_count] * entity_num).to(world.device)
                i2rs[item] = torch.IntTensor([self.relation_count] * entity_num).to(world.device)
        return i2es, i2rs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail
