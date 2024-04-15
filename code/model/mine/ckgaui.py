import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_sum

import model
import world
from train import dataloader, losses, utils


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
        self.ui_gcn_layers = world.hyper_CKGAUI_ui_layers
        self.ui_gcn_graph_dropout = 0.2
        self.ckg_gcn_layers = world.hyper_CKGAUI_ckg_layers

    def prepare_each_epoch(self):
        pass

    def calculate_embedding(self):
        user_ckg_rep, item_ckg_rep = self.ckg_gcn(self.ckg_gcn_layers,
                                                  self.embedding_user.weight,
                                                  torch.concat([self.embedding_item.weight,
                                                                self.embedding_entity.weight]),
                                                  self.inter_edge,
                                                  self.inter_edge_w,
                                                  self.edge_index,
                                                  self.edge_type)

        # UU link prediction，两个数值应该同时变大，更多空间但更难进入
        # uu_topK, uu_threshold = 5, 0.5
        # uu_pred = torch.sigmoid(torch.mm(user_ckg_rep, user_ckg_rep.T)).fill_diagonal_(0)
        # values, col_indices = torch.topk(uu_pred, uu_topK, dim=1)  # 梯度断开
        # non_zero_indices = values.gt(uu_threshold).nonzero()
        # uu_index = torch.stack([torch.arange(self.n_users).unsqueeze(1).repeat(1, uu_topK).to(world.device).view(-1)[
        #                             non_zero_indices[:, 0]], col_indices.view(-1)[non_zero_indices[:, 0]]], dim=1).t()
        # uu_value = (values[non_zero_indices[:, 0], non_zero_indices[:, 1]] - uu_threshold) / (1 - uu_threshold)
        # uu_value = (self.inter_edge_w.mean() - self.inter_edge_w.min()) * uu_value + self.inter_edge_w.min()
        # inter_edge_w = self.inter_edge_w * (self.inter_edge_w.sum() / (self.inter_edge_w.sum() + uu_value.sum()))

        # II link prediction
        # ii_topK, ii_threshold, ii_weight_threshold = 5, 0.75, 0.1
        # ii_pred = torch.sigmoid(torch.mm(item_ckg_rep[:self.n_items], item_ckg_rep[:self.n_items].T)).fill_diagonal_(0)
        # values, col_indices = torch.topk(ii_pred, ii_topK, dim=1)  # 梯度断开
        # non_zero_indices = values.gt(ii_threshold).nonzero()
        # ii_index = torch.stack([torch.arange(self.n_items).unsqueeze(1).repeat(1, ii_topK).to(world.device).view(-1)[
        #                             non_zero_indices[:, 0]], col_indices.view(-1)[non_zero_indices[:, 0]]], dim=1).t()
        # random_value = torch.rand(non_zero_indices.size(0)).to(world.device)
        # ii_value = ii_weight_threshold * random_value

        # UI GCN
        # self.Graph = utils.construct_graph(self.inter_edge, self.inter_edge_w)
        # self.Graph = utils.construct_graph(self.inter_edge, inter_edge_w, uu_index, uu_value)
        # self.Graph = utils.construct_graph(self.inter_edge, self.inter_edge_w, ii_index+self.n_users, ii_value)
        self.Graph = utils.construct_graph(self.inter_edge, self.inter_edge_w)
        user_ui_rep, item_ui_rep = self.ui_gcn(self.embedding_user.weight,
                                               self.embedding_item.weight,
                                               self.Graph,
                                               n_layers=self.ui_gcn_layers,
                                               drop_prob=self.ui_gcn_graph_dropout)

        # user_final_rep = torch.concat([user_ui_rep, user_ckg_rep])
        # item_final_rep = torch.concat([item_ui_rep, item_ckg_rep[:self.n_items, :]])
        user_final_rep = user_ui_rep + user_ckg_rep
        item_final_rep = item_ui_rep + item_ckg_rep[:self.n_items, :]
        # user_final_rep = user_ckg_rep
        # item_final_rep = item_ckg_rep[:self.n_items, :]
        # utils.MAD_metric(torch.cat([user_ui_rep, item_ui_rep]))  # TOD
        if not self.training:
            return user_final_rep, item_final_rep
        else:
            return user_final_rep, item_final_rep, user_ui_rep, item_ui_rep, user_ckg_rep, item_ckg_rep

    def calculate_loss(self, users, pos, neg):
        (user_final_rep, item_final_rep, user_ui_rep,
         item_ui_rep, user_ckg_rep, item_ckg_rep) = self.calculate_embedding()

        loss = {losses.Loss.BPR.value: losses.loss_BPR(user_final_rep, item_final_rep, users, pos, neg),
                losses.Loss.Regulation.value: losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                     pos, neg),
                # 跨CKG和UI的SSM
                # losses.Loss.SSL.value: losses.loss_SSM_origin(user_ui_rep, item_ckg_rep, users,
                #                                               pos) * world.hyper_SSM_Regulation/2 +
                #                        losses.loss_SSM_origin(user_ckg_rep, item_ui_rep, users,
                #                                               pos) * world.hyper_SSM_Regulation/2,
                # CKG和UI分别做SSM
                # losses.Loss.SSL.value: losses.loss_SSM_origin(user_ui_rep, item_ui_rep, users,
                #                                               pos) * world.hyper_SSM_Regulation / 2 +
                #                        losses.loss_SSM_origin(user_ckg_rep, item_ckg_rep, users,
                #                                               pos) * world.hyper_SSM_Regulation / 2,
                # 只有UI的SSM
                losses.Loss.SSL.value: losses.loss_SSM_origin(user_ui_rep, item_ui_rep, users,
                                                              pos) * world.hyper_SSM_Regulation,
                # losses.Loss.SSL.value: (losses.loss_info_nce(user_ui_rep, user_ckg_rep, users) +
                #                         losses.loss_info_nce(item_ui_rep, item_ckg_rep[:self.n_items], pos)),
                # losses.Loss.MAE.value: self.create_emb_mae_loss(user_ckg_rep, item_ckg_rep,
                #                                                 user_ui_rep, item_ui_rep,
                #                                                 users, pos),
                # losses.Loss.MAE.value: self.create_inter_mae_loss(user_ckg_rep, item_ckg_rep,
                #                                                   user_ui_rep, item_ui_rep,
                #                                                   users, pos)
                }
        return loss

    @staticmethod
    def create_emb_mae_loss(user_emb, item_emb, user_2_emb, item_2_emb, users, pos):
        source_emb = user_emb[users]
        target_emb = user_2_emb[users]
        scores = -torch.log(torch.sigmoid(torch.mul(source_emb, target_emb).sum(1))).sum()
        source_emb = item_emb[pos]
        target_emb = item_2_emb[pos]
        scores += -torch.log(torch.sigmoid(torch.mul(source_emb, target_emb).sum(1))).sum()
        return scores * world.hyper_ssl_reg

    @staticmethod
    def create_inter_mae_loss(user_emb, item_emb, user_2_emb, item_2_emb, users, pos):
        inter_1 = torch.mul(user_emb[users], item_emb[pos])
        inter_2 = torch.mul(user_2_emb[users], item_2_emb[pos])
        scores = -torch.log(torch.sigmoid(torch.mul(inter_1, inter_2).sum(1))).sum()
        return scores * world.hyper_ssl_reg
