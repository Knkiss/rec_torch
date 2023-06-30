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


class KGCN(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.device = world.device
        self.kg_dataset = dataloader.KGDataset()
        self.n_entities = self.kg_dataset.entity_count + self.n_items
        self.n_relations = self.kg_dataset.relation_count

        # load parameters info
        self.embedding_size = world.embedding_dim
        # number of iterations when computing entity representation
        self.n_iter = 1
        self.aggregator_class = 'sum'  # which aggregator to use
        self.reg_weight = 1e-7  # weight of l2 regularization
        self.neighbor_sample_size = 4

        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = torch.nn.Embedding(self.n_relations + 1, self.embedding_size)

        # sample neighbors
        kg_graph = self.kg_dataset.get_kg_graph()
        adj_entity, adj_relation = self.construct_adj(kg_graph)
        self.adj_entity, self.adj_relation = adj_entity.to(self.device), adj_relation.to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.n_iter):
            self.linear_layers.append(
                torch.nn.Linear(
                    self.embedding_size
                    if not self.aggregator_class == "concat"
                    else self.embedding_size * 2,
                    self.embedding_size,
                )
            )
        self.ReLU = torch.nn.ReLU()
        self.Tanh = torch.nn.Tanh()

        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.l2_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["adj_entity", "adj_relation"]

    def get_neighbors(self, items):
        items = torch.unsqueeze(items, dim=1)
        entities = [items]
        relations = []
        for i in range(self.n_iter):
            index = torch.flatten(entities[i])
            neighbor_entities = torch.index_select(self.adj_entity, 0, index).reshape(self.batch_size, -1)
            neighbor_relations = torch.index_select(self.adj_relation, 0, index).reshape(self.batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            user_embeddings = user_embeddings.reshape(self.batch_size, 1, 1, self.embedding_size)
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
            user_relation_scores_normalized = self.softmax(user_relation_scores)
            user_relation_scores_normalized = torch.unsqueeze(user_relation_scores_normalized, dim=-1)
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, dim=2)
        else:
            neighbors_aggregated = torch.mean(neighbor_vectors, dim=2)
        return neighbors_aggregated

    def aggregate(self, user_embeddings, entities, relations):
        entity_vectors = [self.entity_embedding(i) for i in entities]
        relation_vectors = [self.relation_embedding(i) for i in relations]
        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (self.batch_size, -1, self.neighbor_sample_size, self.embedding_size,)
                self_vectors = entity_vectors[hop]
                neighbor_vectors = entity_vectors[hop + 1].reshape(shape)
                neighbor_relations = relation_vectors[hop].reshape(shape)
                neighbors_agg = self.mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
                if self.aggregator_class == "sum":
                    output = (self_vectors + neighbors_agg).reshape(-1, self.embedding_size)
                elif self.aggregator_class == "neighbor":
                    output = neighbors_agg.reshape(-1, self.embedding_size)
                elif self.aggregator_class == "concat":
                    output = torch.cat([self_vectors, neighbors_agg], dim=-1)
                    output = output.reshape(-1, self.embedding_size * 2)
                else:
                    raise Exception("Unknown aggregator: " + self.aggregator_class)
                output = self.linear_layers[i](output)
                output = output.reshape(self.batch_size, -1, self.embedding_size)
                if i == self.n_iter - 1:
                    vector = self.Tanh(output)
                else:
                    vector = self.ReLU(output)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        item_embeddings = entity_vectors[0].reshape(self.batch_size, self.embedding_size)
        return item_embeddings

    def forward(self, user, item):
        self.batch_size = item.shape[0]
        user_e = self.user_embedding(user)
        entities, relations = self.get_neighbors(item)
        item_e = self.aggregate(user_e, entities, relations)
        return user_e, item_e

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def construct_adj(self, kg_graph):
        r"""Get neighbors and corresponding relations for each entity in the KG.

        Args:
            kg_graph(scipy.sparse.coo_matrix): an undirected graph

        Returns:
            tuple:
                - adj_entity(torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                  shape: [n_entities, neighbor_sample_size]
                - adj_relation(torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                  shape: [n_entities, neighbor_sample_size]
        """
        # self.logger.info('constructing knowledge graph ...')
        # treat the KG as an undirected graph
        kg_dict = dict()
        for triple in zip(kg_graph.row, kg_graph.data, kg_graph.col):
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if head not in kg_dict:
                kg_dict[head] = []
            kg_dict[head].append((tail, relation))
            if tail not in kg_dict:
                kg_dict[tail] = []
            kg_dict[tail].append((head, relation))

        # self.logger.info('constructing adjacency matrix ...')
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
        entity_num = kg_graph.shape[0]
        adj_entity = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        for entity in range(entity_num):
            if entity not in kg_dict.keys():
                adj_entity[entity] = np.array([entity] * self.neighbor_sample_size)
                adj_relation[entity] = np.array([0] * self.neighbor_sample_size)
                continue

            neighbors = kg_dict[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=False,
                )
            else:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=True,
                )
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return torch.from_numpy(adj_entity), torch.from_numpy(adj_relation)

    def calculate_loss(self, users, pos, neg):
        loss = {}

        user_e, pos_item_e = self.forward(users.long(), pos.long())
        user_e, neg_item_e = self.forward(users.long(), neg.long())

        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        predict = torch.cat((pos_item_score, neg_item_score))
        target = torch.zeros(len(users) * 2, dtype=torch.float32).to(self.device)
        target[: len(users)] = 1
        rec_loss = self.bce_loss(predict, target)

        l2_loss = self.l2_loss(user_e, pos_item_e, neg_item_e)
        loss[Loss.BPR.value] = rec_loss + self.reg_weight * l2_loss
        return loss

    def calculate_embedding(self):
        return NotImplementedError("不应该运行到此处")

    def getUsersRating(self, users):
        # 每epoch可能运行一次
        # 计算batch用户的得分

        user_index = torch.Tensor(users)
        item_index = torch.tensor(range(self.n_items)).to(self.device)
        item_index = torch.split(item_index, world.test_u_batch_size)

        score = []
        for i in item_index:
            user = torch.unsqueeze(user_index, dim=1).repeat(1, i.shape[0])
            user = torch.flatten(user)
            item = torch.unsqueeze(i, dim=0).repeat(user_index.shape[0], 1)
            item = torch.flatten(item)
            users_emb, items_emb = self.forward(user, item)
            score.append(torch.mul(users_emb, items_emb).sum(dim=1).reshape(user_index.shape[0], -1))

        rating = torch.cat(score, dim=1)
        return rating
