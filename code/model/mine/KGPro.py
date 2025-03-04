import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax
import scipy.sparse as sp

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


class KGPro(model.AbstractRecModel):
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

        self.ui_layers = 3  # Not use
        self.ckg_layers = 2  # default=2

        # 增强UI图，每个epoch更新一次
        self.Graph_enhanced = None
        # 随机门控MLP，每个epoch更新一次
        self.noise_gating = None
        # 可学习门控MLP
        self.adaptive_gating = nn.Linear(self.embedding_dim, self.embedding_dim).to(world.device)
        nn.init.normal_(self.adaptive_gating.weight, std=0.1, mean=0.0)

    def prepare_each_epoch(self):
        # # 噪声gating的参数随机初始化
        # self.noise_gating = nn.Linear(self.embedding_dim, self.embedding_dim).to(world.device)
        # nn.init.normal_(self.noise_gating.weight, std=0.1, mean=0.0)

        # 从ckg上采样边，补充到UI上，构成UI增强图
        eu, ei = self.embedding_user.weight, self.embedding_item.weight
        zu_g0, zi_g0 = self.ui_gcn(eu, ei, self.Graph)
        zu_g1, zi_g1 = self.ckg_gcn(self.ckg_layers,
                                    self.embedding_user.weight,
                                    torch.concat([self.embedding_item.weight, self.embedding_entity.weight]),
                                    self.inter_edge,
                                    self.inter_edge_w,
                                    self.edge_index,
                                    self.edge_type)
        # 计算UI和CKG的交互图
        zi_g1 = zi_g1[:self.n_items]
        ui_predict_graph = torch.mm(zu_g0, zi_g0.T).cpu()
        ckg_predict_graph = torch.mm(zu_g1, zi_g1.T).cpu()
        diff_predict = ckg_predict_graph - ui_predict_graph
        # diff_predict = ckg_predict_graph  # TODO Ablation w/o E_ui

        # 所有正值归一化到0-1，作为边的权重
        diff_predict[diff_predict < 0] = 0
        # 非线性归一化 TODO 超参数 tau温度系数
        tau = 1.5    # 5=0.3939 1.5=0.3947 1=0.3946 0.75=3941 0.5=0.3925
        normalized_diff_predict = torch.sigmoid(torch.exp(diff_predict / tau))
        # 线性归一化
        # normalized_diff_predict = (diff_predict - diff_predict.min()) / (diff_predict.max() - diff_predict.min())

        # 每个用户找出前5个边，作为可选的prompt_edge TODO 超参数 topk
        topk = 20  # 50=0.2912 25=0.3930 20=0.3973 5=0.3947
        values, cols = torch.topk(normalized_diff_predict, topk, dim=1)
        values = values.reshape(-1)
        cols = cols.reshape(-1)
        rows = torch.arange(normalized_diff_predict.shape[0], device=world.device).repeat(topk, 1).T.reshape(-1)

        # CKG prompt边的加权比例，TODO 超参数beta 补充比例
        beta = 0.05  # 0.01=0.3947 0.05=0.3964 0.1=0.3962 0.5=0.2909
        values = beta * values

        # 构造增强UI图
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        ori_values = np.ones(len(self.ui_dataset.trainUser))
        ori_rows = self.ui_dataset.trainUser
        ori_cols = self.ui_dataset.trainItem

        stack_values = np.concatenate([ori_values, values.cpu().detach().numpy()])
        stack_rows = np.concatenate([ori_rows, rows.cpu().detach().numpy()])
        stack_cols = np.concatenate([ori_cols, cols.cpu().detach().numpy()])
        new_UserItemNet = sp.csr_matrix((stack_values, (stack_rows, stack_cols)),
                                        shape=(self.n_users, self.n_items))
        # new_UserItemNet = sp.csr_matrix((values.cpu().detach().numpy(),
        #                                  (rows.cpu().detach().numpy(),
        #                                   cols.cpu().detach().numpy())),
        #                                 shape=(self.n_users, self.n_items))  # TODO ablation w/o G_ui

        R = new_UserItemNet.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        self.Graph_enhanced = utils.convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph_enhanced = self.Graph_enhanced.coalesce().to(world.device)

    def calculate_embedding(self, enhanced_ui_graph=False):
        eu, ei = self.embedding_user.weight, self.embedding_item.weight

        # embedding加扰动
        all_emb = torch.concat([eu, ei], dim=0)
        # noise_ratio = torch.sigmoid(self.noise_gating(all_emb))
        # noise_all_emb = torch.mul(all_emb, noise_ratio)
        noise_all_emb = torch.nn.functional.dropout(all_emb, p=0.2, training=True)  # gating和noise的区别
        noise_eu, noise_ei = torch.split(noise_all_emb, [self.n_users, self.n_items], dim=0)
        # TODO Ablation w/o RG
        # noise_eu, noise_ei = torch.split(all_emb, [self.n_users, self.n_items], dim=0)

        # 增强图的前向计算
        layer_num = 2  # TODO 超参数 增强图传播层数 1=0.3935 2=0.3974 3=0.3959 4=0.2893
        enhanced_zu_g0, enhanced_zi_g0 = self.ui_gcn(noise_eu, noise_ei, self.Graph_enhanced, n_layers=layer_num)
        enhanced_all_emb = torch.concat([enhanced_zu_g0, enhanced_zi_g0], dim=0)

        # 可学习mlp的回传计算
        final_all_emb = all_emb + torch.mul(enhanced_all_emb, torch.sigmoid(self.adaptive_gating(enhanced_all_emb)))
        # TODO Ablation w/o AG
        # final_all_emb = all_emb + enhanced_all_emb

        # 最终计算
        final_eu, final_ei = torch.split(final_all_emb, [self.n_users, self.n_items], dim=0)
        zu_g0, zi_g0 = self.ui_gcn(final_eu, final_ei, self.Graph)  # 推荐

        # TODO Ablation w/o PL
        # zu_g0, zi_g0 = torch.split(enhanced_all_emb, [self.n_users, self.n_items], dim=0)

        zu_g1, zi_g1 = self.ckg_gcn(self.ckg_layers,  # CKG优化
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
            return zu_g0, zi_g0

    def calculate_loss(self, users, pos, neg):
        # 噪声gating的参数随机初始化
        self.noise_gating = nn.Linear(self.embedding_dim, self.embedding_dim).to(world.device)
        nn.init.normal_(self.noise_gating.weight, std=0.1, mean=0.0)

        eu, ei, g0 = self.embedding_user.weight, self.embedding_item.weight, self.Graph
        zu, zi, zu_g0, zu_g1, zi_g0, zi_g1 = self.calculate_embedding()

        loss = dict()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_g0, zi_g0, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(eu, ei, users, pos, neg)
        loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g1, zi_g1, users, pos)
        return loss
