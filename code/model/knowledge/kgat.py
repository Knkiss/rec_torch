from torch.nn import Module
from typing import Callable, TypeVar

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from scipy.sparse import coo_matrix

import model
import world
from train import dataloader
from train.losses import Loss

T = TypeVar('T', bound='Module')


class KGAT(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.device = world.device

        self.kg_dataset = dataloader.KGDataset()
        self.n_entities = self.kg_dataset.entity_count + self.n_items
        self.n_relations = self.kg_dataset.relation_count

        # load dataset info
        ckg_coo = self._create_ckg_sparse_matrix(self.ui_dataset.UserItemNet, self.kg_dataset.get_kg_graph())
        self.ckg = self._create_ckg_graph(self.ui_dataset.UserItemNet, self.kg_dataset.get_kg_graph())

        self.all_hs = torch.LongTensor(ckg_coo.row).to(self.device)
        self.all_ts = torch.LongTensor(ckg_coo.col).to(self.device)
        self.all_rs = torch.LongTensor(ckg_coo.data).to(self.device)
        self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])

        # load parameters info
        self.embedding_size = world.hyper_embedding_dim
        self.kg_embedding_size = world.hyper_embedding_dim
        self.layers = world.hyper_KGAT_layers
        self.aggregator_type = 'bi'
        self.mess_dropout = 0.1
        self.reg_weight = 1e-5

        # generate intermediate data
        self.A_in = (self.init_graph())  # init the attention matrix by the structure of ckg
        self.A_in.requires_grad = False

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = torch.nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = torch.nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.aggregator_layers = torch.nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))
        self.tanh = torch.nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(xavier_normal_initialization)

    def forward(self):
        return self.calculate_embedding()

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _create_ckg_sparse_matrix(self, ui_matrix, kg_matrix):
        ui_matrix = ui_matrix.tocoo()
        user_num = self.n_users

        hids = kg_matrix.row + user_num
        tids = kg_matrix.col + user_num
        uids = ui_matrix.row
        iids = ui_matrix.col + user_num
        src = np.concatenate([uids, iids, hids])
        tgt = np.concatenate([iids, uids, tids])

        ui_rel_num = len(uids)
        ui_rel_id = self.n_relations - 1
        kg_rel = kg_matrix.data
        ui_rel = np.full(2 * ui_rel_num, ui_rel_id, dtype=kg_rel.dtype)
        data = np.concatenate([ui_rel, kg_rel])

        node_num = self.n_entities + self.n_users
        mat = coo_matrix((data, (src, tgt)), shape=(node_num, node_num))
        return mat

    def _create_ckg_graph(self, ui_matrix, kg_matrix):
        ui_matrix = ui_matrix.tocoo()
        user_num = self.n_users

        hids = kg_matrix.row + user_num
        tids = kg_matrix.col + user_num
        uids = ui_matrix.row
        iids = ui_matrix.col + user_num
        src = np.concatenate([uids, iids, hids])
        tgt = np.concatenate([iids, uids, tids])

        ui_rel_num = len(uids)
        ui_rel_id = self.n_relations - 1
        kg_rel = torch.Tensor(kg_matrix.data)
        ui_rel = torch.full((2 * ui_rel_num,), ui_rel_id, dtype=kg_rel.dtype)
        edge = torch.cat([ui_rel, kg_rel])

        import dgl
        graph = dgl.graph((src, tgt))
        graph.edata["relation_id"] = edge
        return graph

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl

        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            col, row = dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False).adj().coo()
            col, row = col.numpy(), row.numpy()
            data = np.ones(col.shape[0]).astype(np.int64)
            sub_graph = (sp.coo_matrix(
                (data, (col, row)), shape=(self.ckg.number_of_nodes(), self.ckg.number_of_nodes()), dtype=np.float32))
            rowsum = np.array(sub_graph.sum(1))
            # 源代码有无穷大运行时警告，chatgpt修改
            # 源代码：d_inv = np.power(rowsum, -1).flatten()
            d_inv = np.where(rowsum != 0, np.power(rowsum, -1), 0).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor(np.array([final_adj_matrix.row, final_adj_matrix.col]))
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(world.device)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        user_all_embeddings, entity_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[users.long()]
        pos_embeddings = entity_all_embeddings[pos.long()]
        neg_embeddings = entity_all_embeddings[neg.long()]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        loss[Loss.BPR.value] = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_embedding(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = torch.nn.functional.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings[:self.num_items]

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def calculate_loss_transE(self, h, r, pos_t, neg_t):
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = torch.nn.functional.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss
        return loss

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def generate_transE_score(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(
            self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix"""

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in.data = A_in

    def prepare_each_epoch(self):
        self.update_attentive_A()


class Aggregator(torch.nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = torch.nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = torch.nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = torch.nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = torch.nn.Linear(self.input_dim, self.output_dim)
            self.W2 = torch.nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = torch.nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings
