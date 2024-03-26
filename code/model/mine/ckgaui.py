import math
from collections import defaultdict
from os.path import join

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_sum
from tqdm import tqdm

import model
import world
from train import dataloader, losses
from train.losses import Loss

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def read_triplets(file_name, inverse_r):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np[:, 2] = can_triplets_np[:, 2] + n_items
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    edge_index_sampled = None
    edge_type_sampled = None
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat([edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat([edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _edge_sampling(edge_index, edge_type, samp_rate):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(torch.Tensor(random_tensor)).type(torch.bool)
    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate
    return i, v


def _get_edges(graph):
    graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
    index = graph_tensor[:, :-1]  # [-1, 2]
    type = graph_tensor[:, -1]  # [-1, 1]
    return index.t().long().to(world.device), type.long().to(world.device)


def _convert_sp_mat_to_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return i.to(world.device), v.to(world.device)


class AttnHGCN(nn.Module):
    def __init__(self, channel, n_relations, mess_dropout_rate):
        super(AttnHGCN, self).__init__()
        self.relation_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_relations, channel)))

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))
        nn.init.xavier_uniform_(self.W_Q)

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    # def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
    #     n_entities = entity_emb.shape[0]
    #     head, tail = edge_index
    #
    #     # attention from entity to item/entity
    #     query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
    #     key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)
    #     key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
    #     edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
    #     edge_attn_score = scatter_softmax(edge_attn_score, head)
    #
    #     relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
    #     neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
    #     value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
    #     entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
    #     entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
    #     entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
    #     item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
    #     user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
    #
    #     return entity_agg, user_agg
    #
    # def forward_ckg(self, layer_nums, user_emb, entity_emb, edge_index, edge_type,
    #                 inter_edge, inter_edge_w, mess_dropout=True, item_attn=None):
    #     if item_attn is not None:
    #         item_attn = item_attn[inter_edge[1, :]]
    #         item_attn = scatter_softmax(item_attn, inter_edge[0, :])
    #         norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
    #         norm = torch.index_select(norm, 0, inter_edge[0, :])
    #         item_attn = item_attn * norm
    #         inter_edge_w = inter_edge_w * item_attn
    #
    #     entity_res_emb = entity_emb
    #     user_res_emb = user_emb
    #     for i in range(layer_nums):
    #         entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge,
    #                                                      inter_edge_w, self.relation_emb)
    #
    #         entity_emb = F.normalize(self.dropout(entity_emb) if mess_dropout else entity_emb)
    #         user_emb = F.normalize(self.dropout(user_emb) if mess_dropout else user_emb)
    #
    #         entity_res_emb = torch.add(entity_res_emb, entity_emb)
    #         user_res_emb = torch.add(user_res_emb, user_emb)
    #
    #     return entity_res_emb, user_res_emb

    def forward_ui(self, layers_num, user_emb, item_emb, inter_edge, inter_edge_w):
        item_res_emb = [item_emb]
        user_res_emb = [user_emb]

        if self.training:
            inter_edge, inter_edge_w = _sparse_dropout(inter_edge, inter_edge_w, 0.8)

        for i in range(layers_num):
            item_agg = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
            user_agg = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]

            user_emb = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
            item_emb = scatter_sum(src=user_agg, index=inter_edge[1, :], dim_size=item_emb.shape[0], dim=0)

            item_res_emb.append(item_emb)
            user_res_emb.append(user_emb)

        item_res_emb = torch.mean(torch.stack(item_res_emb, dim=1), dim=1)
        user_res_emb = torch.mean(torch.stack(user_res_emb, dim=1), dim=1)

        return item_res_emb, user_res_emb


class CKGAUI(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.kg_dataset = dataloader.KGDataset()

        # count
        self.n_relations = self.kg_dataset.relation_count * 2 - 1  # double relations
        self.n_entities = self.kg_dataset.entity_count - 1 + self.n_items
        self.n_nodes = self.n_entities + self.n_users

        # prepare CKG data
        # dataset_path = join(world.PATH_DATA, world.dataset)
        # train_cf = read_cf(dataset_path + '/train.txt')
        # test_cf = read_cf(dataset_path + '/test.txt')
        # remap_item(train_cf, test_cf)
        # triplets = read_triplets(dataset_path + '/kg.txt', inverse_r=True)
        # graph, relation_dict = build_graph(train_cf, triplets)
        # _, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)
        # _, self.inter_edge_mean = _convert_sp_mat_to_tensor(mean_mat_list[0])  # 交互矩阵

        self.inter_edge_w = self.Graph.values()[:self.Graph.values().shape[0] // 2]
        self.inter_edge = torch.stack([self.Graph.indices()[0, :self.Graph.indices()[0].shape[0] // 2],
                                       self.Graph.indices()[1, :self.Graph.indices()[0].shape[0] // 2] - self.n_users],
                                      dim=0)

        # self.edge_index, self.edge_type = _get_edges(graph)  # 知识图谱
        # self.all_embed = nn.init.normal_(torch.empty(self.n_users+self.n_items, self.embedding_dim), std=0.1)
        # self.all_embed = nn.Parameter(self.all_embed)
        # self.embedding_entity = torch.nn.Embedding(self.n_entities - self.n_items, self.embedding_dim)
        # nn.init.normal_(self.embedding_entity.weight, std=0.1)

        # 分开创建torch.nn.Embedding，共同初始化
        # self.embedding_user = torch.nn.Embedding(self.num_users, self.embedding_dim)
        # self.embedding_item = torch.nn.Embedding(self.num_items, self.embedding_dim)
        # self.embedding_entity = torch.nn.Embedding(self.n_entities-self.n_items, self.embedding_dim)
        # all_embeddings = torch.cat([self.embedding_user.weight, self.embedding_item.weight, self.embedding_entity.weight], dim=0)
        # nn.init.normal_(all_embeddings, std=0.1)
        # self.embedding_user.weight.data.copy_(all_embeddings[:self.n_users])
        # self.embedding_item.weight.data.copy_(all_embeddings[self.n_users:self.n_users+self.n_items])
        # self.embedding_entity.weight.data.copy_(all_embeddings[self.n_users+self.n_items:])

        # 分开创建torch.nn.Embedding，分开初始化
        # self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        # self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        # self.embedding_entity = torch.nn.Embedding(num_embeddings=self.n_entities-self.n_items, embedding_dim=self.embedding_dim)
        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_item.weight, std=0.1)
        # nn.init.normal_(self.embedding_entity.weight, std=0.1)

        # 共同创建torch.Parameter，共同优化
        self.all_embed = nn.init.normal_(torch.empty(self.n_nodes, self.embedding_dim), std=0.1)
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = AttnHGCN(channel=self.embedding_dim,
                            n_relations=self.n_relations,
                            mess_dropout_rate=0.2)

    def calculate_embedding(self):
        # entity_gcn_emb, user_gcn_emb = self.gcn.forward_ckg(user_emb,
        #                                                     item_emb,
        #                                                     self.edge_index,
        #                                                     self.edge_type,
        #                                                     self.inter_edge,
        #                                                     self.inter_edge_w,
        #                                                     mess_dropout=False)

        # user_emb = self.embedding_user.weight
        # item_emb = self.embedding_item.weight
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb = self.gcn.forward_ui(3, user_emb, item_emb, self.inter_edge, self.inter_edge_w)
        return user_gcn_emb, entity_gcn_emb

    def calculate_loss(self, users, pos, neg):
        # edge_index, edge_type = _relation_aware_edge_sampling(self.edge_index, self.edge_type,
        #                                                       self.n_relations, self.node_dropout_rate)
        #
        # # rec task
        # entity_gcn_emb, user_gcn_emb = self.gcn.forward_ckg(user_emb,
        #                                                     item_emb,
        #                                                     edge_index,
        #                                                     edge_type,
        #                                                     inter_edge,
        #                                                     inter_edge_w,
        #                                                     mess_dropout=self.mess_dropout)
        # user_emb = self.embedding_user.weight
        # item_emb = self.embedding_item.weight
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb = self.gcn.forward_ui(3, user_emb, item_emb, self.inter_edge, self.inter_edge_w)

        loss = {}
        loss[losses.Loss.BPR.value] = losses.loss_BPR(user_gcn_emb, entity_gcn_emb, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(user_emb, item_emb, users, pos, neg)
        return loss
