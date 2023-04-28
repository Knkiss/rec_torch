import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

import model
import world
from train import losses, dataloader, utils


class GraphADD(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities + 1,
                                                   embedding_dim=self.embedding_dim)
        self.embedding_relation = torch.nn.Embedding(num_embeddings=self.num_relations + 1,
                                                     embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)

        self.model = model.LightGCN()

    def calculate_embedding(self):
        return self.model(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                    pos, neg)
        return loss

    def prepare_each_epoch(self):
        pass
        # TODO 使用KG 计算 attention_graph_add 【M+N,M+N】
        user_entity = torch.mm(self.embedding_user.weight, self.embedding_entity.weight.T)
        item_entity = torch.mm(self.embedding_item.weight, self.embedding_entity.weight.T)

        user_item = torch.mm(user_entity, item_entity.T)
        user_item = (user_item - user_item.min()) / (user_item.max() - user_item.min())

        topk_values, topk_indices = torch.topk(user_item.view(-1), k=100000)
        item_mask = torch.zeros_like(user_item)
        item_mask.view(-1)[topk_indices] = 1

        self.change_graph(item_mask)

    def change_graph(self, item_mask):
        m_items = self.ui_dataset.m_items
        n_users = self.ui_dataset.n_users
        R = self.ui_dataset.UserItemNet

        csr_matrix = sp.csr_matrix(item_mask.cpu().numpy())
        R = R + csr_matrix
        R.data = np.where(R.data > 1, 1, R.data)

        R = R.tolil()
        adj_mat = sp.dok_matrix((n_users + m_items, n_users + m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:n_users, n_users:] = R
        adj_mat[n_users:, :n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        self.Graph = utils.convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(world.device)
        self.Graph.requires_grad = False
