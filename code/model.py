# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：model.py
@Author  ：Knkiss
@Date    ：2023/2/14 12:01 
"""
import torch.nn.functional as F
from torch import nn

from GAT import GAT
import dataloader
from dataloader import *
from utils import _L2_loss_mean


class KGCL(nn.Module):
    def __init__(self, config: dict):
        super(KGCL, self).__init__()
        self.config = config
        self.ui_dataset = dataloader.UIDataset()
        self.kg_dataset = dataloader.KGDataset()

        self.num_users = self.ui_dataset.n_users
        self.num_items = self.ui_dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities + 1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(num_embeddings=self.num_relations + 1,
                                                     embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)

        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.f = nn.Sigmoid()
        self.Graph = self.ui_dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()
        print(f"KGCL is ready to go!")

    @staticmethod
    def __dropout_x(x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def lightGCN_drop(self, g_droped, kg_droped):
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(kg_droped)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def lightGCN(self):
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout'] and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.lightGCN()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.lightGCN()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if torch.isnan(loss).any().tolist():
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        r_embed = self.embedding_relation(r)
        h_embed = self.embedding_item(h)
        pos_t_embed = self.embedding_entity(pos_t)
        neg_t_embed = self.embedding_entity(neg_t)

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
            neg_t_embed)
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def cal_item_embedding_rgat(self, kg: dict):
        item_embs = self.embedding_item(torch.LongTensor(list(kg.keys())).to(world.device))  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.embedding_entity(item_entities.long())  # item_num, entity_num_each, emb_dim
        relation_embs = self.embedding_relation(item_relations.long())  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities,torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict
        return self.cal_item_embedding_rgat(kg)

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.lightGCN()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class Baseline(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.ui_dataset = dataloader.UIDataset(path=join(world.DATA_PATH, world.dataset))
        self.num_users = self.ui_dataset.n_users
        self.num_items = self.ui_dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.Graph = self.ui_dataset.getSparseGraph()
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()

    def model_MF(self):
        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight
        return all_users, all_items

    def model_lightGCN(self):
        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight
        all_emb = torch.cat([all_users, all_items])
        embs = [all_emb]
        if self.config['dropout'] and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        all_users, all_items = torch.split(light_out, [self.num_users, self.num_items])
        return all_users, all_items

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    @staticmethod
    def __dropout_x(x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def select_model(self):
        if world.model == 'MF':
            return self.model_MF()
        elif world.model == 'lightGCN':
            return self.model_lightGCN()

    def bpr_loss(self, users, pos, neg):
        all_users, all_items = self.select_model()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]
        userEmb0 = self.embedding_user(users.long())
        posEmb0 = self.embedding_item(pos.long())
        negEmb0 = self.embedding_item(neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if torch.isnan(loss).any().tolist():
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss

    def getUsersRating(self, users):
        all_users, all_items = self.select_model()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
