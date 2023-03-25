# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：model.py
@Author  ：Knkiss
@Date    ：2023/2/14 12:01 
"""
import torch.nn.functional as F
from torch import nn

import dataloader
import module
from dataloader import *
from main import Loss


class AbstractRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = world.config
        self.ui_dataset = dataloader.UIDataset()
        self.num_users = self.ui_dataset.n_users
        self.num_items = self.ui_dataset.m_items
        self.Graph = self.ui_dataset.getSparseGraph()

        self.latent_dim = self.config['latent_dim_rec']

        if world.pretrain_input_enable:
            emb = torch.load(world.pretrain_folder + world.dataset + '_' + world.pretrain_input + '.pretrain')
            self.embedding_user = torch.nn.Embedding.from_pretrained(emb['embedding_user.weight'])
            self.embedding_item = torch.nn.Embedding.from_pretrained(emb['embedding_item.weight'])
            self.embedding_user.requires_grad_()
            self.embedding_item.requires_grad_()
        else:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()

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


class Baseline(AbstractRecModel):
    def __init__(self):
        super().__init__()
        if world.model == 'lightGCN':
            self.model = module.LightGCN()

    def calculate_loss(self, users, pos, neg):
        losses = {}
        all_users, all_items = self.calculate_embedding()
        losses[Loss.BPR.value] = utils.loss_BPR(all_users, all_items, users, pos, neg)
        losses[Loss.Regulation.value] = utils.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        return losses

    def calculate_embedding(self):
        if world.model == 'MF':
            return self.embedding_user.weight, self.embedding_item.weight
        elif world.model == 'lightGCN':
            return self.model(self.embedding_user.weight, self.embedding_item.weight, self.Graph)


class KGCL(AbstractRecModel):
    def __init__(self):
        super(KGCL, self).__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))
        self.n_layers = self.config['lightGCN_n_layers']

        self.contrast_views = {}

        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities + 1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(num_embeddings=self.num_relations + 1,
                                                     embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)

        self.lightGCN = module.LightGCN()

        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.Graph = self.ui_dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)
        self.gat = module.GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()
        print(f"KGCL is ready to go!")

    def ui_drop_weighted(self, item_mask):
        item_mask = item_mask.tolist()
        n_nodes = self.num_users + self.num_items
        item_np = self.ui_dataset.trainItem
        keep_idx = list()
        if world.user_item_preference:
            for i, j in enumerate(item_mask):
                if j:
                    keep_idx.append(i)
        else:
            for i, j in enumerate(item_np.tolist()):
                if item_mask[j]:
                    keep_idx.append(i)

        # print(f"finally keep ratio: {len(keep_idx) / len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        user_np = self.ui_dataset.trainUser[keep_idx]
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(world.device)
        g.requires_grad = False
        return g

    def get_ui_views_weighted(self, item_stabilities, stab_weight):
        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        kg_weights = (item_stabilities - item_stabilities.min()) / (item_stabilities.max() - item_stabilities.min())
        kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = (1 - world.ui_p_drop) / torch.mean(stab_weight * kg_weights) * (stab_weight * kg_weights)
        weights = weights.where(weights < 0.95, torch.ones_like(weights) * 0.95)

        item_mask = torch.bernoulli(weights).to(torch.bool)
        # print(f"keep ratio: {item_mask.sum() / item_mask.size()[0]:.2f}")
        # drop
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        return g_weighted

    def item_kg_stability(self, view1, view2):
        kgv1_ro = self.cal_item_embedding_from_kg(view1)  # items * dims
        kgv2_ro = self.cal_item_embedding_from_kg(view2)  # items * dims
        if world.user_item_preference:
            userList = torch.LongTensor(self.ui_dataset.trainUser).to(world.device)
            user1_emb = self.embedding_user(userList)  # inters * dims
            user2_emb = self.embedding_user(userList)  # inters * dims
            item1_emb = kgv1_ro[self.ui_dataset.trainItem]  # inters * dim
            item2_emb = kgv2_ro[self.ui_dataset.trainItem]  # inters * dim
            ui1_emb = torch.cat((user1_emb, item1_emb), dim=1)  # inters * dim*2
            ui2_emb = torch.cat((user2_emb, item2_emb), dim=1)  # inters * dim*2
            sim = F.cosine_similarity(ui1_emb, ui2_emb)  # inters 交互数量
        else:
            sim = utils.sim(kgv1_ro, kgv2_ro)  # items 物品数量
        return sim

    def get_views(self):
        kgv1, kgv2 = self.get_kg_views()
        stability = self.item_kg_stability(kgv1, kgv2).to(world.device)
        uiv1 = self.get_ui_views_weighted(stability, 1)
        uiv2 = self.get_ui_views_weighted(stability, 1)

        contrast_views = {
            "kgv1": kgv1,
            "kgv2": kgv2,
            "uiv1": uiv1,
            "uiv2": uiv2
        }
        return contrast_views

    def get_kg_views(self):
        if world.item_entity_random_walk:
            kg, _ = self.kg_dataset.get_kg_dict_random(self.num_items)
        else:
            kg = self.kg_dict
        view1 = utils.drop_edge_random(kg, world.kg_p_drop, self.num_entities)
        view2 = utils.drop_edge_random(kg, world.kg_p_drop, self.num_entities)
        return view1, view2

    def lightGCN_drop(self, g_droped, kg_droped):
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(kg_droped)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def prepare_each_epoch(self):
        self.contrast_views = self.get_views()

    def calculate_embedding(self):
        return self.lightGCN(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        losses = {}
        all_users, all_items = self.lightGCN(self.embedding_user.weight, self.embedding_item.weight, self.Graph)
        losses[Loss.BPR.value] = utils.loss_BPR(all_users, all_items, users, pos, neg)
        losses[Loss.Regulation.value] = utils.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        uiv1, uiv2 = self.contrast_views["uiv1"], self.contrast_views["uiv2"]
        kgv1, kgv2 = self.contrast_views["kgv1"], self.contrast_views["kgv2"]
        usersV1_ro, itemsV1_ro = self.lightGCN_drop(uiv1, kgv1)
        usersV2_ro, itemsV2_ro = self.lightGCN_drop(uiv2, kgv2)
        loss_ssl_item = utils.loss_info_nce(itemsV1_ro, itemsV2_ro, pos)
        loss_ssl_user = utils.loss_info_nce(usersV1_ro, usersV2_ro, users)
        losses[Loss.SSL.value] = loss_ssl_user + loss_ssl_item
        return losses

    def calculate_loss_transE(self, h, r, pos_t, neg_t):
        loss = utils.loss_transE(self.embedding_item,
                                 self.embedding_entity,
                                 self.embedding_relation,
                                 h, r, pos_t, neg_t)
        return loss

    def cal_item_embedding_from_kg(self, kg: dict):
        item_embs = self.embedding_item(torch.LongTensor(list(kg.keys())).to(world.device))  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.embedding_entity(item_entities.long())  # item_num, entity_num_each, emb_dim
        relation_embs = self.embedding_relation(item_relations.long())  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)


class SGL(AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.graph_1 = None
        self.graph_2 = None
        if world.model == 'SGL':
            self.model = module.LightGCN()
        elif world.model == 'GraphCL':
            self.model = QKV()

    def create_adj_mat(self, is_subgraph=True, aug_type=0):
        training_user = self.ui_dataset.trainUser
        training_item = self.ui_dataset.trainItem

        n_nodes = self.num_users + self.num_items
        if is_subgraph and aug_type in [0, 1, 2] and world.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = utils.randint_choice(self.num_users, size=int(self.num_users * world.ssl_ratio),
                                                     replace=False)
                drop_item_idx = utils.randint_choice(self.num_items, size=int(self.num_items * world.ssl_ratio),
                                                     replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(training_user, dtype=np.float32), (training_user, training_item)),
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = utils.randint_choice(len(training_user), size=int(len(training_user) * (1 - world.ssl_ratio)),
                                                replace=False)
                user_np = np.array(training_user)[keep_idx]
                item_np = np.array(training_item)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(training_user)
            item_np = np.array(training_item)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        Graph = utils.convert_sp_mat_to_sp_tensor(adj_matrix)
        Graph = Graph.coalesce().to(world.device)
        return Graph

    def prepare_each_epoch(self):
        self.graph_1 = self.create_adj_mat(True, aug_type=1)
        self.graph_2 = self.create_adj_mat(True, aug_type=1)

    def calculate_embedding_graph(self, all_users, all_items, graph):
        if world.model == 'SGL' or world.model == 'GraphCL':
            all_users, all_items = self.model(all_users, all_items, graph)
        return all_users, all_items

    def calculate_embedding(self):
        return self.calculate_embedding_graph(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        losses = {}
        all_users, all_items = self.calculate_embedding()
        users_1, items_1 = self.calculate_embedding_graph(self.embedding_user.weight,
                                                          self.embedding_item.weight, self.graph_1)
        users_2, items_2 = self.calculate_embedding_graph(self.embedding_user.weight,
                                                          self.embedding_item.weight, self.graph_2)
        losses[Loss.BPR.value] = utils.loss_BPR(all_users, all_items, users, pos, neg)
        losses[Loss.Regulation.value] = utils.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        loss_ssl_item = utils.loss_info_nce(items_1, items_2, pos)
        loss_ssl_user = utils.loss_info_nce(users_1, users_2, users)
        losses[Loss.SSL.value] = loss_ssl_user + loss_ssl_item
        return losses


class QKV(AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.lightGCN = module.LightGCN()
        self.QGrouping = module.QGrouping()

    def calculate_embedding(self):
        return self.forward(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        losses = {}
        all_users, all_items = self.calculate_embedding()
        losses[Loss.BPR.value] = utils.loss_BPR(all_users, all_items, users, pos, neg)
        losses[Loss.Regulation.value] = utils.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        return losses

    def forward(self, all_users, all_items, graph):
        all_users, all_items = self.lightGCN(all_users, all_items, graph)
        if not world.QKV_only_user:
            all_users, all_items = self.QGrouping(all_users), self.QGrouping(all_items)
            all_users = torch.reshape(all_users, shape=[self.num_users, self.QGrouping.v_dim * self.QGrouping.q_dim])
            all_items = torch.reshape(all_items, shape=[self.num_items, self.QGrouping.v_dim * self.QGrouping.q_dim])
        else:
            all_users = self.QGrouping(all_users)
            all_users = torch.mean(all_users, dim=2)
        return all_users, all_items
