import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax

import model
import world
from train import losses, dataloader, utils


class CKGGCN(nn.Module):
    def __init__(self, dims, n_relations):
        super(CKGGCN, self).__init__()
        self.relation_emb = nn.Parameter(nn.init.normal_(torch.empty(n_relations, dims), std=0.1))

        self.W_Q = nn.Parameter(nn.init.normal_(torch.Tensor(dims, dims), std=0.1))

        self.n_heads = 2
        self.d_k = dims // self.n_heads

    def _agg_layer(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        head, tail = edge_index
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]

        if world.hyper_WORK2_ablation_model == 4:
            entity_agg = tail_emb * self.relation_emb[edge_type - 1]
        else:
            # attention from entity to item/entity
            query = (head_emb @ self.W_Q).view(-1, self.n_heads, self.d_k)
            key = (tail_emb @ self.W_Q).view(-1, self.n_heads, self.d_k)
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
            edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
            edge_attn_score = scatter_softmax(edge_attn_score, head)
            relation_emb = self.relation_emb[edge_type - 1]
            neigh_relation_emb = tail_emb * relation_emb  # [-1, channel]
            value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
            entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)

        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        entity_agg_res = torch.zeros_like(entity_emb)
        entity_agg = entity_agg_res.index_add_(0, head, entity_agg)
        entity_agg = F.normalize(entity_agg)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = torch.zeros_like(user_emb)
        user_agg = user_agg.index_add_(0, inter_edge[0, :], item_agg)
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


class WORK2(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.n_relations = self.kg_dataset.relation_count * 2 - 1
        self.num_entities = self.kg_dataset.entity_count - 1  # exclude items
        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities, embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)

        self.inter_edge_w = self.Graph.values()[:self.Graph.values().shape[0] // 2]
        self.inter_edge = [self.Graph.indices()[0, :self.Graph.indices()[0].shape[0] // 2],
                           self.Graph.indices()[1, :self.Graph.indices()[0].shape[0] // 2] - self.n_users]
        self.inter_edge = torch.stack(self.inter_edge, dim=0)

        self.Graph_KG = self.kg_dataset.get_kg_graph(1, True)
        self.edge_type = torch.LongTensor(self.Graph_KG.data).to(world.device)
        self.edge_index = torch.LongTensor(np.stack((self.Graph_KG.row, self.Graph_KG.col))).to(world.device)

        self.ui_gcn = model.LightGCN()
        self.ckg_gcn = CKGGCN(dims=self.embedding_dim, n_relations=self.n_relations)

        self.ui_layers = 3
        self.ckg_layers = 2

        # self.matrix_resample = MatrixResample(self.ui_dataset,
        #                                       self.Graph_KG,
        #                                       self.n_users,
        #                                       self.n_items,
        #                                       self.num_entities)
        self.Graph_resample = self.Graph  # Prepare each epoch

        # Use for KD on item group
        self.group_mlp = nn.Linear(in_features=self.embedding_dim,
                                   out_features=world.hyper_WORK2_cluster_num).to(world.device)
        nn.init.normal_(self.group_mlp.weight, std=0.1)

        # if world.hyper_WORK2_reset_ui_graph:
        #     self.Graph_resample = self.matrix_resample.prepare_init_from_pretrain()

    # def prepare_each_epoch(self):
    #     eu, ei, ee, g0 = (self.embedding_user.weight,
    #                       self.embedding_item.weight,
    #                       self.embedding_entity.weight,
    #                       self.Graph)
    #     zu_ui, zi_ui = self.ui_gcn(eu, ei, g0)
    #     zu_ckg, zi_ckg = self.ckg_gcn(self.ckg_layers,
    #                                   eu,
    #                                   torch.concat([ei, ee]),
    #                                   self.inter_edge,
    #                                   self.inter_edge_w,
    #                                   self.edge_index,
    #                                   self.edge_type)
    #
    #     self.matrix_resample.prepare_each_epoch(zu_ckg, zi_ckg[:self.n_items])

    def prepare_each_epoch(self):
        if world.hyper_WORK2_ablation_model == 3:
            self.graph_1 = utils.create_adj_mat(self.ui_dataset.trainUser, self.ui_dataset.trainItem,
                                                self.num_users, self.num_items, is_subgraph=True)

    def calculate_embedding(self):
        eu, ei = self.embedding_user.weight, self.embedding_item.weight
        zu_g0, zi_g0 = self.ui_gcn(eu, ei, self.Graph_resample)
        if world.hyper_WORK2_ablation_model == 3:
            zu_g1, zi_g1 = self.ui_gcn(eu, ei, self.graph_1)
        else:
            zu_g1, zi_g1 = self.ckg_gcn(self.ckg_layers,
                                        self.embedding_user.weight,
                                        torch.concat([self.embedding_item.weight, self.embedding_entity.weight]),
                                        self.inter_edge,
                                        self.inter_edge_w,
                                        self.edge_index,
                                        self.edge_type)
            zi_g1 = zi_g1[:self.n_items]

        if self.training:
            return zu_g0 + zu_g1, zi_g0 + zi_g1, zu_g0, zu_g1, zi_g0, zi_g1
        else:
            if world.hyper_WORK2_BPR_mode == 1:
                return zu_g0, zi_g0
            elif world.hyper_WORK2_BPR_mode == 2:
                return zu_g1, zi_g1
            else:
                return zu_g0 + zu_g1, zi_g0 + zi_g1

    def calculate_loss(self, users, pos, neg):
        eu, ei, g0 = self.embedding_user.weight, self.embedding_item.weight, self.Graph
        zu, zi, zu_g0, zu_g1, zi_g0, zi_g1 = self.calculate_embedding()

        loss = dict()
        if world.hyper_WORK2_BPR_mode == 1:
            loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_g0, zi_g0, users, pos, neg)
        elif world.hyper_WORK2_BPR_mode == 2:
            loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_g1, zi_g1, users, pos, neg)
        else:
            loss[losses.Loss.BPR.value] = losses.loss_BPR(zu, zi, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(eu, ei, users, pos, neg)

        if (world.hyper_WORK2_SSM_mode > 0
                or world.hyper_WORK2_ablation_model == 2
                or world.hyper_WORK2_ablation_model == 5):
            if world.hyper_WORK2_ablation_model == 2:
                loss[losses.Loss.SSL.value] = losses.loss_BPR(zu_g1, zi_g1, users, pos, neg)
            elif world.hyper_WORK2_ablation_model == 5:
                loss[losses.Loss.SSL.value] = (losses.loss_info_nce(zi_g0, zi_g1, pos) +
                                               losses.loss_info_nce(zu_g0, zu_g1, users))
                return loss
            elif world.hyper_WORK2_SSM_mode == 1:
                loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g0, zi_g0, users, pos)
            elif world.hyper_WORK2_SSM_mode == 2:
                loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g1, zi_g1, users, pos)
            elif world.hyper_WORK2_SSM_mode == 3:
                loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu, zi, users, pos)
            else:
                raise NotImplementedError("world.hyper_WORK2_SSM_mode")

        if world.hyper_WORK2_KD_mode > 0 and world.hyper_WORK2_ablation_model != 1:
            if world.hyper_WORK2_KD_mode == 1:
                loss[losses.Loss.MAE.value] = losses.loss_kd_ii_graph_batch(zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 2:
                loss[losses.Loss.MAE.value] = losses.loss_kd_ii_graph_batch(zi_g1, zi_g0, pos, neg)  # w/ neg
            elif world.hyper_WORK2_KD_mode == 3:
                loss[losses.Loss.MAE.value] = losses.loss_kd_cluster_ii_graph_batch(zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 4:
                loss[losses.Loss.MAE.value] = losses.loss_kd_cluster_ii_graph_batch(zi_g1, zi_g0)
            elif world.hyper_WORK2_KD_mode == 5:
                loss[losses.Loss.MAE.value] = losses.loss_kd_A_graph_batch(zu_g1, zu_g0, zi_g1, zi_g0, users, pos)
            elif world.hyper_WORK2_KD_mode == 6:
                loss[losses.Loss.MAE.value] = losses.loss_kd_mlp_ii_graph_batch(self.group_mlp, zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 7:
                loss[losses.Loss.MAE.value] = losses.loss_kd_mlp_ii_graph_batch(self.group_mlp, zi_g1, zi_g0)
            elif world.hyper_WORK2_KD_mode == 8:
                loss[losses.Loss.MAE.value] = losses.loss_bpr_mlp_ui_graph_batch(self.group_mlp, zi_g1, zi_g0,
                                                                                 zu_g1, zu_g0, users, pos, neg,
                                                                                 form='BPR')
            elif world.hyper_WORK2_KD_mode == 9:
                loss[losses.Loss.MAE.value] = losses.loss_bpr_mlp_ui_graph_batch(self.group_mlp, zi_g1, zi_g0,
                                                                                 zu_g1, zu_g0, users, pos, neg,
                                                                                 form='InfoNCE')
            elif world.hyper_WORK2_KD_mode == 10:
                loss[losses.Loss.MAE.value] = losses.loss_mlp_cluster_contrastive(self.group_mlp, zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 11:
                loss[losses.Loss.MAE.value] = losses.loss_mlp_cluster_contrastive(self.group_mlp, zi_g1, zi_g0, pos,
                                                                                  reverse=True)
            else:
                raise NotImplementedError("world.hyper_WORK2_KD_mode")

        return loss
