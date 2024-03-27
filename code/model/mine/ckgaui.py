import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_sum

import model
import world
from train import dataloader, losses


class CKGGCN(nn.Module):
    def __init__(self, dims, n_relations):
        super(CKGGCN, self).__init__()
        self.relation_emb = nn.Parameter(nn.init.normal_(torch.empty(n_relations, dims), std=0.1))

        self.W_Q = nn.Parameter(nn.init.normal_(torch.Tensor(dims, dims), std=0.1))

        self.n_heads = 2
        self.d_k = dims // self.n_heads

    def _agg_layer(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        # attention from entity to item/entity
        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        relation_emb = self.relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
        entity_agg = F.normalize(entity_agg)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)

        return entity_agg, user_agg

    def forward(self, layers_num, user_emb, entity_emb, inter_edge, inter_edge_w, edge_index, edge_type):
        user_embs = [user_emb]
        entity_embs = [entity_emb]
        for i in range(layers_num):
            entity_emb, user_emb = self._agg_layer(user_emb, entity_emb,
                                                   edge_index, edge_type,
                                                   inter_edge, inter_edge_w)
            user_embs.append(user_emb)
            entity_embs.append(entity_emb)
        user_embs = torch.mean(torch.stack(user_embs, dim=1), dim=1)
        entity_embs = torch.mean(torch.stack(entity_embs, dim=1), dim=1)
        return user_embs, entity_embs


class CKGAUI(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        # Data prepared
        self.kg_dataset = dataloader.KGDataset()
        self.n_relations = self.kg_dataset.relation_count * 2 - 1
        self.n_entities = self.kg_dataset.entity_count - 1 + self.n_items
        self.n_nodes = self.n_entities + self.n_users
        self.Graph_KG = self.kg_dataset.get_kg_graph(1, True)
        self.edge_type = torch.LongTensor(self.Graph_KG.data).to(world.device)
        self.edge_index = torch.LongTensor(np.stack((self.Graph_KG.row, self.Graph_KG.col))).to(world.device)

        self.inter_edge_w = self.Graph.values()[:self.Graph.values().shape[0] // 2]
        self.inter_edge = [self.Graph.indices()[0, :self.Graph.indices()[0].shape[0] // 2],
                           self.Graph.indices()[1, :self.Graph.indices()[0].shape[0] // 2] - self.n_users]
        self.inter_edge = torch.stack(self.inter_edge, dim=0)

        # Models parameters
        self.embedding_entity = torch.nn.Embedding(self.n_entities - self.n_items, self.embedding_dim)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        self.ckg_gcn = CKGGCN(dims=self.embedding_dim, n_relations=self.n_relations)
        self.ui_gcn = model.LightGCN()

        # Hyperparameters
        self.ui_gcn_layers = 3
        self.ui_gcn_graph_dropout = 0.2
        self.ckg_gcn_layers = 2

    def calculate_embedding(self):
        # user_ui_rep, item_ui_rep = self.ui_gcn(self.embedding_user.weight,
        #                                        self.embedding_item.weight,
        #                                        utils.construct_graph(self.inter_edge, self.inter_edge_w),
        #                                        n_layers=self.ui_gcn_layers,
        #                                        drop_prob=self.ui_gcn_dropout)
        user_ckg_rep, item_ckg_rep = self.ckg_gcn(self.ckg_gcn_layers,
                                                  self.embedding_user.weight,
                                                  torch.concat([self.embedding_item.weight,
                                                                self.embedding_entity.weight]),
                                                  self.inter_edge,
                                                  self.inter_edge_w,
                                                  self.edge_index,
                                                  self.edge_type)

        # user_final_rep, item_final_rep = user_ui_rep, item_ui_rep
        user_final_rep, item_final_rep = user_ckg_rep, item_ckg_rep
        return user_final_rep, item_final_rep

    def calculate_loss(self, users, pos, neg):
        # user_ui_rep, item_ui_rep = self.ui_gcn(self.embedding_user.weight,
        #                                        self.embedding_item.weight,
        #                                        utils.construct_graph(self.inter_edge, self.inter_edge_w),
        #                                        n_layers=self.ui_gcn_layers,
        #                                        drop_prob=self.ui_gcn_dropout)
        user_ckg_rep, item_ckg_rep = self.ckg_gcn(self.ckg_gcn_layers,
                                                  self.embedding_user.weight,
                                                  torch.concat([self.embedding_item.weight,
                                                                self.embedding_entity.weight]),
                                                  self.inter_edge,
                                                  self.inter_edge_w,
                                                  self.edge_index,
                                                  self.edge_type)

        # user_final_rep, item_final_rep = user_ui_rep, item_ui_rep
        user_final_rep, item_final_rep = user_ckg_rep, item_ckg_rep

        loss = {losses.Loss.BPR.value: losses.loss_BPR(user_final_rep, item_final_rep, users, pos, neg),
                losses.Loss.Regulation.value: losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                     pos, neg)}
        # loss_ssl_item = self.create_mae_loss(item_ckg_rep[:self.n_items, :], item_ui_rep)
        # loss_ssl_user = self.create_mae_loss(user_ckg_rep, user_ui_rep)
        # loss[losses.Loss.SSL.value] = loss_ssl_user + loss_ssl_item
        return loss

    # @staticmethod
    # def create_mae_loss(source_emb, target_emb):
    #     scores = -torch.log(torch.sigmoid(torch.mul(source_emb, target_emb).sum(1))).sum()
    #     return scores * 0.01
