import torch
from torch import nn

import world
from train import dataloader

class AbstractRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        if world.root_model:
            return
        world.root_model = True
        self.ui_dataset = dataloader.UIDataset()
        self.num_users = self.ui_dataset.n_users
        self.num_items = self.ui_dataset.m_items
        self.Graph = self.ui_dataset.getSparseGraph()

        self.embedding_dim = world.embedding_dim
        if world.pretrain_input_enable:
            emb = torch.load(world.PRETRAIN_PATH + '/' + world.dataset + '_' + world.pretrain_input + '.pretrain')
            self.embedding_user = torch.nn.Embedding.from_pretrained(emb['embedding_user.weight'])
            self.embedding_item = torch.nn.Embedding.from_pretrained(emb['embedding_item.weight'])
            self.embedding_user.requires_grad_()
            self.embedding_item.requires_grad_()
        else:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()

    @property
    def n_users(self):
        return self.num_users

    @property
    def n_items(self):
        return self.num_items

    def prepare_each_epoch(self):
        # 每epoch运行一次
        # 例如：对比学习需要在每个epoch计算不同视图下的多种emb
        # 可以通过此函数预先在batch计算外进行处理
        pass

    def calculate_embedding(self):
        # 每batch计算一次
        # 在完整的图上计算用户和物品的emb，用于BPR或预测计算
        raise NotImplementedError

    def calculate_loss(self, users, pos, neg):
        # 每batch计算一次
        # 计算模型的全部Loss，并以minibatch进行优化
        raise NotImplementedError

    def getUsersRating(self, users):
        # 每epoch可能运行一次
        # 计算batch用户的得分
        all_users, all_items = self.calculate_embedding()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating