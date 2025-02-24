import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax

import model
import world
from train import losses, dataloader, utils
import scipy.sparse as sp


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

        # Add aggregation for the signal from user to item (reverse direction)
        user_to_item_agg = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg_res = torch.zeros_like(entity_emb)
        item_agg_res = item_agg_res.index_add_(0, inter_edge[1, :], user_to_item_agg)
        entity_agg += item_agg_res

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


class WORK3(model.AbstractRecModel):
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
        self.ckg_layers = 3

        # 初始化用于在新graph上传播的embedding
        self.n_prompt = 10
        self.embedding_prompt = torch.nn.Embedding(num_embeddings=self.n_prompt, embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_prompt.weight, std=0.1)

        # 在每个epoch开始时更新，额外使用embedding_prompt用于推荐任务
        self.construct_edges_ratio = 0.03
        self.Graph_augmented = None

    def prepare_each_epoch(self):
        self.Graph_augmented = self.construct_graph()
        # self.Graph_augmented = None

    def construct_graph(self):
        # 使用CKG通道引导UI通道的子图构建，额外补充prompt_node节点的连接关系
        zu_ui, zi_ui, zu_ckg, zi_ckg = self.calculate_embedding()
        zi_ui = zi_ui[:self.num_items]
        prompt_node = self.embedding_prompt.weight

        z_ckg = torch.concat([zu_ckg, zi_ckg], dim=0)
        connect_ckg = torch.mm(z_ckg, prompt_node.t())
        z_ui = torch.concat([zu_ui, zi_ui], dim=0)
        connect_ui = torch.mm(z_ui, prompt_node.t())
        diff_connect = connect_ckg - connect_ui
        positive_edges = diff_connect > 0
        positive_edges_idx = positive_edges.nonzero(as_tuple=True)  # 获取符合条件的边的索引

        # top比例
        # connect_values = diff_connect[positive_edges_idx]
        # top_k = int(positive_edges_idx[0].size(0) * self.construct_edges_ratio)
        # _, top_k_indices = torch.topk(connect_values, top_k, largest=True)
        # sampled_edges = (positive_edges_idx[0][top_k_indices], positive_edges_idx[1][top_k_indices])

        # sample比例
        sample_size = int(positive_edges_idx[0].size(0) * self.construct_edges_ratio)
        random_indices = torch.randint(0, positive_edges_idx[0].size(0), (sample_size,))
        sampled_edges = (positive_edges_idx[0][random_indices], positive_edges_idx[1][random_indices])

        sampled_edges_0_cpu = sampled_edges[0].cpu().numpy()
        sampled_edges_1_cpu = sampled_edges[1].cpu().numpy()
        print(len(sampled_edges_0_cpu))
        add_csr = sp.csr_matrix(
            (torch.ones(len(sampled_edges_0_cpu)), (sampled_edges_0_cpu, sampled_edges_1_cpu)),
            shape=(self.num_users+self.num_items, self.n_prompt)  # 确保矩阵的形状与self.Graph一致
        )

        adj_mat = sp.dok_matrix((self.num_users + self.num_items + self.n_prompt,
                                 self.num_users + self.num_items + self.n_prompt), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.ui_dataset.UserItemNet.tolil()
        adj_mat[:self.num_users, self.num_users:-self.n_prompt] = R
        adj_mat[self.num_users:-self.n_prompt, :self.num_users] = R.T
        A = add_csr.tolil()
        adj_mat[:-self.n_prompt, -self.n_prompt:] = A
        adj_mat[-self.n_prompt:, :-self.n_prompt] = A.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph_augmented = utils.convert_sp_mat_to_sp_tensor(norm_adj)
        Graph_augmented = Graph_augmented.coalesce().to(world.device)

        return Graph_augmented

    def calculate_embedding(self):
        eu, ei = self.embedding_user.weight, self.embedding_item.weight

        if self.training and self.Graph_augmented is not None:
            concat_emb = torch.concat([ei, self.embedding_prompt.weight], dim=0)
            zu_ui, zi_ui = self.ui_gcn(eu, concat_emb, self.Graph_augmented)
        else:
            zu_ui, zi_ui = self.ui_gcn(eu, ei, self.Graph)

        zu_ckg, zi_ckg = self.ckg_gcn(self.ckg_layers,
                                      self.embedding_user.weight,
                                      torch.concat([self.embedding_item.weight, self.embedding_entity.weight]),
                                      self.inter_edge,
                                      self.inter_edge_w,
                                      self.edge_index,
                                      self.edge_type)
        zi_ckg = zi_ckg[:self.n_items]

        if self.training:
            return zu_ui, zi_ui, zu_ckg, zi_ckg
        else:
            return zu_ui, zi_ui

    def calculate_loss(self, users, pos, neg):
        eu, ei, g0 = self.embedding_user.weight, self.embedding_item.weight, self.Graph
        zu_ui, zi_ui, zu_ckg, zi_ckg = self.calculate_embedding()

        loss = dict()

        # UI侧
        loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_ui, zi_ui, users, pos, neg)
        # loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_ui, zi_ui, users, pos)
        # loss[losses.Loss.BPR.value] = losses.loss_knowledge_bpr(zu_ui, zi_ui, zu_ckg, zi_ckg, users, pos, neg)

        # CKG侧
        # loss[losses.Loss.SSL.value] = losses.loss_BPR(zu_ckg, zi_ckg, users, pos, neg)
        loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_ckg, zi_ckg, users, pos)
        # loss[losses.Loss.BPR.value] = losses.loss_knowledge_bpr(zu_ui, zi_ui, zu_ckg, zi_ckg, users, pos, neg)
        # loss[losses.Loss.SSL.value] = losses.loss_knowledge_SSM_origin(zu_ui, zi_ui, zu_ckg, zi_ckg, users, pos)

        loss[losses.Loss.Regulation.value] = losses.loss_regulation(eu, ei, users, pos, neg)
        return loss
