from typing import Callable, TypeVar

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss

import model
import world
from train import dataloader
from train.losses import Loss

T = TypeVar('T', bound='Module')


class KGIN(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.n_entities = self.kg_dataset.entity_count + self.n_items
        self.n_relations = self.kg_dataset.relation_count

        self.embedding_size = world.embedding_dim
        self.n_factors = 4
        self.context_hops = 2
        self.node_dropout_rate = 0.5
        self.mess_dropout_rate = 0.0
        self.ind = 'cosine'
        self.sim_decay = 1e-4
        self.reg_weight = 1e-5
        self.temperature = 0.2

        self.inter_matrix = self.ui_dataset.UserItemNet.tocoo().astype(np.float32)  # [n_users, n_items]
        # inter_matrix: [n_users, n_entities]; inter_graph: [n_users + n_entities, n_users + n_entities]
        self.interact_mat, _ = self.get_norm_inter_matrix(mode="si")

        self.kg_graph = self.kg_dataset.get_kg_graph()  # [n_entities, n_entities]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)  # edge_index: [2, -1]; edge_type: [-1,]

        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = torch.nn.Embedding(self.n_factors, self.embedding_size)

        self.gcn = GraphConv(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_factors=self.n_factors,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            interact_mat=self.interact_mat,
            ind=self.ind,
            tmp=self.temperature,
            device=world.device,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(xavier_uniform_initialization)

    def forward(self):
        return self.calculate_embedding()

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(world.device), type.to(world.device)

    def get_norm_inter_matrix(self, mode="bi"):
        # Get the normalized interaction matrix of users and items.
        def _bi_norm_lap(A):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(A.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(A):
            # D^{-1}A
            rowsum = np.array(A.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(A)
            return norm_adj.tocoo()

        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=np.float32,
        )
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        if mode == "bi":
            L = _bi_norm_lap(A)
        elif mode == "si":
            L = _si_norm_lap(A)
        else:
            raise NotImplementedError(
                f"Normalize mode [{mode}] has not been implemented."
            )
        # covert norm_inter_graph to tensor
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        norm_graph = torch.sparse.FloatTensor(i, data, L.shape)

        # interaction: user->item, [n_users, n_entities]
        L_ = L.tocsr()[: self.n_users, self.n_users:].tocoo()
        # covert norm_inter_matrix to tensor
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        norm_matrix = torch.sparse.FloatTensor(i_, data_, L_.shape)

        return norm_matrix.to(world.device), norm_graph.to(world.device)

    def calculate_loss(self, users, pos, neg):
        loss = {}

        user_all_embeddings, entity_all_embeddings, cor_loss = self.forward()
        u_embeddings = user_all_embeddings[users.long()]
        pos_embeddings = entity_all_embeddings[pos.long()]
        neg_embeddings = entity_all_embeddings[neg.long()]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        cor_loss = self.sim_decay * cor_loss
        loss[Loss.BPR.value] = mf_loss + self.reg_weight * reg_loss + cor_loss

        return loss

    def calculate_embedding(self):
        user_embeddings = self.user_embedding.weight # user_gcn_emb: [n_users, embedding_size]
        entity_embeddings = self.entity_embedding.weight # entity_gcn_emb: [n_entities, embedding_size]
        latent_embeddings = self.latent_embedding.weight # latent_gcn_emb: [n_factors, embedding_size]

        entity_gcn_emb, user_gcn_emb, cor_loss = self.gcn(
            user_embeddings, entity_embeddings, latent_embeddings
        )

        if self.training:
            return user_gcn_emb, entity_gcn_emb[:self.n_items], cor_loss
        else:
            return user_gcn_emb, entity_gcn_emb[:self.n_items]


class Aggregator(torch.nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(
        self,
    ):
        super(Aggregator, self).__init__()

    def forward(
        self,
        entity_emb,
        user_emb,
        latent_emb,
        relation_emb,
        edge_index,
        edge_type,
        interact_mat,
        disen_weight_att,
    ):
        from torch_scatter import scatter_mean

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type]
        neigh_relation_emb = (
            entity_emb[tail] * edge_relation_emb
        )  # [-1, embedding_size]
        entity_agg = scatter_mean(
            src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0
        )

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = torch.nn.Softmax(dim=1)(score_)  # [n_users, n_factors]
        """user aggregate"""
        user_agg = torch.sparse.mm(
            interact_mat, entity_emb
        )  # [n_users, embedding_size]
        disen_weight = torch.mm(
            torch.nn.Softmax(dim=-1)(disen_weight_att), relation_emb
        )  # [n_factors, embedding_size]
        user_agg = (
            torch.mm(score, disen_weight)
        ) * user_agg + user_agg  # [n_users, embedding_size]

        return entity_agg, user_agg


class GraphConv(torch.nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(
        self,
        embedding_size,
        n_hops,
        n_users,
        n_factors,
        n_relations,
        edge_index,
        edge_type,
        interact_mat,
        ind,
        tmp,
        device,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1,
    ):
        super(GraphConv, self).__init__()

        self.embedding_size = embedding_size
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.interact_mat = interact_mat
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.temperature = tmp
        self.device = device

        # define layers
        self.relation_embedding = torch.nn.Embedding(self.n_relations, self.embedding_size)
        disen_weight_att = torch.nn.init.xavier_uniform_(torch.empty(n_factors, n_relations))
        self.disen_weight_att = torch.nn.Parameter(disen_weight_att)
        self.convs = torch.nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = torch.nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, user_emb, entity_emb, latent_emb):
        """node dropout"""
        # node dropout
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate
            )
            interact_mat = self.node_dropout(self.interact_mat)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat

        entity_res_emb = entity_emb  # [n_entities, embedding_size]
        user_res_emb = user_emb  # [n_users, embedding_size]
        relation_emb = self.relation_embedding.weight  # [n_relations, embedding_size]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](
                entity_emb,
                user_emb,
                latent_emb,
                relation_emb,
                edge_index,
                edge_type,
                interact_mat,
                self.disen_weight_att,
            )
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = torch.nn.functional.normalize(entity_emb)
            user_emb = torch.nn.functional.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return (
            entity_res_emb,
            user_res_emb,
            self.calculate_cor_loss(self.disen_weight_att),
        )

    def calculate_cor_loss(self, tensors):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = torch.nn.functional.normalize(tensor_1, dim=0)
            normalized_tensor_2 = torch.nn.functional.normalize(tensor_2, dim=0)
            return (normalized_tensor_1 * normalized_tensor_2).sum(
                dim=0
            ) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = (
                torch.matmul(tensor_1, tensor_1.t()) * 2,
                torch.matmul(tensor_2, tensor_2.t()) * 2,
            )  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
            a, b = torch.sqrt(
                torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8
            ), torch.sqrt(
                torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8
            )  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel**2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel**2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel**2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation(tensors):
            # tensors: [n_factors, dimension]
            # normalized_tensors: [n_factors, dimension]
            normalized_tensors = torch.nn.functional.normalize(tensors, dim=1)
            scores = torch.mm(normalized_tensors, normalized_tensors.t())
            scores = torch.exp(scores / self.temperature)
            cor_loss = -torch.sum(torch.log(scores.diag() / scores.sum(1)))
            return cor_loss

        """cul similarity for each latent factor weight pairs"""
        if self.ind == "mi":
            return MutualInformation(tensors)
        elif self.ind == "distance":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += DistanceCorrelation(tensors[i], tensors[j])
        elif self.ind == "cosine":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += CosineSimilarity(tensors[i], tensors[j])
        else:
            raise NotImplementedError(
                f"The independence loss type [{self.ind}] has not been supported."
            )
        return cor_loss
