import math
from collections import defaultdict
from logging import getLogger
from os.path import join

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_mean, scatter_sum
from tqdm import tqdm

import model
import world
from train import dataloader
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
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat([edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat([edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(torch.Tensor(random_tensor)).type(torch.bool)
    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate
    return i, v


def _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
    _, least_attn_edge_id = torch.topk(-edge_attn_score,
                                       int((1 - keep_rate) * edge_attn_score.shape[0]), sorted=False)
    cl_kg_mask = torch.ones_like(edge_attn_score).bool()
    cl_kg_mask[least_attn_edge_id] = False
    cl_kg_edge = edge_index[:, cl_kg_mask]
    cl_kg_type = edge_type[cl_kg_mask]
    return cl_kg_edge, cl_kg_type


def _adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func="torch"):
    inter_attn_prob = item_attn_mean[inter_edge[1]]
    # add gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
    """ prob based drop """
    inter_attn_prob = inter_attn_prob + noise
    inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

    if samp_func == "np":
        # we observed abnormal behavior of torch.multinomial on mind
        sampled_edge_idx = np.random.choice(np.arange(inter_edge_w.shape[0]),
                                            size=int(keep_rate * inter_edge_w.shape[0]), replace=False,
                                            p=inter_attn_prob.cpu().numpy())
    else:
        sampled_edge_idx = torch.multinomial(inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False)

    return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx] / keep_rate


class AttnHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_relations,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, mess_dropout=True, item_attn=None):

        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge,
                                                         inter_edge_w, self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb

    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # print attn score
        if print:
            self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score


class Contrast(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7):
        super(Contrast, self).__init__()
        self.tau: float = tau

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        neg_sim = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))

        return -torch.log(between_sim / (between_sim + between_sim + neg_sim))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)
        loss = self.loss(h1, h2).mean()
        return loss


class KGRec(model.AbstractRecModel):
    def __init__(self, hp_dict=None):
        super(KGRec, self).__init__()
        self.kg_dataset = dataloader.KGDataset()

        # count
        self.num_entities = self.kg_dataset.entity_count - 1  # exclude items
        self.n_relations = self.kg_dataset.relation_count * 2 - 1
        self.n_entities = self.kg_dataset.entity_count - 1 + self.n_items  # include items
        self.n_nodes = self.n_entities + self.n_users  # n_users + n_entities

        self.decay = 1e-5
        self.emb_size = self.embedding_dim
        self.context_hops = 2
        self.node_dropout = 1
        self.node_dropout_rate = 0.5
        self.mess_dropout = 1
        self.mess_dropout_rate = 0.1
        self.device = world.device

        self.ablation = None

        self.mae_coef = 0.1
        self.mae_msize = 256
        self.cl_coef = 0.001
        self.tau = 0.5
        self.cl_drop = 0.5
        self.samp_func = "torch"

        if world.dataset == 'last-fm':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.01
            self.tau = 1.0
            self.cl_drop = 0.5
        elif world.dataset == 'mind-f':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.1
            self.cl_drop = 0.6
            self.samp_func = "np"
        elif world.dataset == 'alibaba-fashion':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.2
            self.cl_drop = 0.5

        # update hps
        if hp_dict is not None:
            for k, v in hp_dict.items():
                setattr(self, k, v)

        # prepare CKG data
        dataset_path = join(world.PATH_DATA, world.dataset)
        train_cf = read_cf(dataset_path + '/train.txt')
        test_cf = read_cf(dataset_path + '/test.txt')
        remap_item(train_cf, test_cf)
        triplets = read_triplets(dataset_path + '/kg.txt', inverse_r=True)
        graph, relation_dict = build_graph(train_cf, triplets)
        adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)
        adj_mat = mean_mat_list[0]

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(adj_mat)
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = AttnHGCN(channel=self.emb_size,
                            n_hops=self.context_hops,
                            n_users=self.n_users,
                            n_relations=self.n_relations,
                            node_dropout_rate=self.node_dropout_rate,
                            mess_dropout_rate=self.mess_dropout_rate)
        self.contrast_fn = Contrast(self.emb_size, tau=self.tau)

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def calculate_embedding(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                item_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.inter_edge,
                                                self.inter_edge_w,
                                                mess_dropout=False)
        return user_gcn_emb, entity_gcn_emb

    def calculate_loss(self, users, pos, neg):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        """node dropout"""
        # 1. graph sprasification;
        edge_index, edge_type = _relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        # 2. compute rationale scores;
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(
            item_emb, edge_index, edge_type, print=False, return_logits=True)
        # for adaptive UI MAE
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]
        # for adaptive MAE training
        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(
            edge_attn_score, self.mae_msize, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(
            edge_index, edge_type, topk_attn_edge_id)

        inter_edge, inter_edge_w = _sparse_dropout(
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

        # rec task
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                item_emb,
                                                enc_edge_index,
                                                enc_edge_type,
                                                inter_edge,
                                                inter_edge_w,
                                                mess_dropout=self.mess_dropout,
                                                )

        loss = {}
        # Rec Task
        u_e = user_gcn_emb[users]
        pos_e, neg_e = entity_gcn_emb[pos], entity_gcn_emb[neg]
        _, loss[Loss.BPR.value], loss[Loss.Regulation.value] = self.create_bpr_loss(u_e, pos_e, neg_e)

        # MAE Task
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]  # mask_size, channel
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type - 1]
        loss[Loss.MAE.value] = self.mae_coef * self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # CL task
        """adaptive sampling"""
        cl_kg_edge, cl_kg_type = _adaptive_kg_drop_cl(
            edge_index, edge_type, edge_attn_score, keep_rate=1 - self.cl_drop)
        cl_ui_edge, cl_ui_w = _adaptive_ui_drop_cl(
            item_attn_mean, inter_edge, inter_edge_w, 1 - self.cl_drop, samp_func=self.samp_func)
        item_agg_ui = self.gcn.forward_ui(
            user_emb, item_emb[:self.n_items], cl_ui_edge, cl_ui_w)
        item_agg_kg = self.gcn.forward_kg(
            item_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        loss[Loss.SSL.value] = self.cl_coef * self.contrast_fn(item_agg_ui, item_agg_kg)

        return loss

    @staticmethod
    def create_mae_loss(node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        if torch.isnan(mf_loss):
            raise ValueError("nan mf_loss")

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
