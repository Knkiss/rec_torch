from typing import Callable, TypeVar
from torch.nn import Module
import torch.nn.functional as F

import torch
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss

import model
import world
from train import dataloader
from train.losses import Loss

T = TypeVar('T', bound='Module')


class CKE(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.n_entities = self.kg_dataset.entity_count
        self.n_relations = self.kg_dataset.relation_count

        self.embedding_size = world.hyper_embedding_dim
        self.kg_embedding_size = world.hyper_embedding_dim
        self.reg_weights = [1e-2, 1e-2]

        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = torch.nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = torch.nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self):
        return self.calculate_embedding()

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        r_e = F.normalize(r_e, p=2, dim=1)
        h_e = F.normalize(h_e, p=2, dim=1)
        pos_t_e = F.normalize(pos_t_e, p=2, dim=1)
        neg_t_e = F.normalize(neg_t_e, p=2, dim=1)

        return h_e, r_e, pos_t_e, neg_t_e, r_trans_w

    def _get_rec_loss(self, user_e, pos_e, neg_e):
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        rec_loss = self.rec_loss(pos_score, neg_score)
        return rec_loss

    def _get_kg_loss(self, h_e, r_e, pos_e, neg_e):
        pos_tail_score = ((h_e + r_e - pos_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_e) ** 2).sum(dim=1)
        kg_loss = self.kg_loss(neg_tail_score, pos_tail_score)
        return kg_loss

    def calculate_loss_transE(self, h, r, pos_t, neg_t):
        h = h.long()
        r = r.long()
        pos_t = pos_t.long()
        neg_t = neg_t.long()
        h_e, r_e, pos_t_e, neg_t_e, r_trans_w = self._get_kg_embedding(h, r, pos_t, neg_t)
        kg_loss = self._get_kg_loss(h_e, r_e, pos_t_e, neg_t_e)
        reg_loss = self.reg_weights[1] * self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        return kg_loss+reg_loss

    def calculate_loss(self, users, pos, neg):
        user = users.long()
        pos_item = pos.long()
        neg_item = neg.long()
        user_e = self.user_embedding(user)
        pos_item_e = self.item_embedding(pos_item)
        neg_item_e = self.item_embedding(neg_item)
        pos_item_kg_e = self.entity_embedding(pos_item)
        neg_item_kg_e = self.entity_embedding(neg_item)
        pos_item_final_e = pos_item_e + pos_item_kg_e
        neg_item_final_e = neg_item_e + neg_item_kg_e
        rec_loss = self._get_rec_loss(user_e, pos_item_final_e, neg_item_final_e)
        reg_loss = self.reg_weights[0] * self.reg_loss(user_e, pos_item_final_e, neg_item_final_e)
        loss = {
            Loss.BPR.value: rec_loss + reg_loss,
        }
        return loss

    def calculate_embedding(self):
        u_e = self.user_embedding.weight
        i_e = self.item_embedding.weight + self.entity_embedding.weight[:self.n_items]
        return u_e, i_e
