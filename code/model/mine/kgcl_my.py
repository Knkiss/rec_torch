import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F

import model
import world
from train import losses, dataloader, utils
from train.losses import Loss


class KGCL_my(model.AbstractRecModel):
    def __init__(self):
        super(KGCL_my, self).__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        if world.dataset_info_show_enable:
            print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))

        self.contrast_views = {}

        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities + 1,
                                                   embedding_dim=self.embedding_dim)
        self.embedding_relation = torch.nn.Embedding(num_embeddings=self.num_relations + 1,
                                                     embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)

        self.lightGCN = model.LightGCN()

        self.W_R = nn.Parameter(torch.Tensor(self.num_relations+1, 1))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.Graph = self.ui_dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)
        self.gat = model.GAT(self.embedding_dim, self.embedding_dim, dropout=0.4, alpha=0.2).train()
        print(f"KGCL is ready to go!")

    def ui_drop_weighted(self, item_mask):
        item_mask = item_mask.tolist()
        n_nodes = self.num_users + self.num_items
        item_np = self.ui_dataset.trainItem
        keep_idx = list()
        for i, j in enumerate(item_mask):
            if j:
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
        np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
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
        g = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape)).coalesce().to(world.device)
        g.requires_grad = False
        return g

    def get_ui_views_weighted(self, item_stabilities, stab_weight):
        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        kg_weights = (item_stabilities - item_stabilities.min()) / (item_stabilities.max() - item_stabilities.min())
        # if world.para_1 == 0:
        #     kg_weights = torch.exp(kg_weights)
        # kg_weights = kg_weights.where(kg_weights > world.para_2, torch.ones_like(kg_weights) * world.para_2)  # 最小值

        # overall probability of keep
        weights = kg_weights
        # weights = (1 - world.hyper_KGCL_ui_p_drop) / torch.mean(kg_weights) * (kg_weights)
        weights = weights.where(weights < world.para_3, torch.ones_like(weights) * world.para_3)  # 最大值0.95

        item_mask = torch.bernoulli(weights).to(torch.bool)
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        return g_weighted

    def item_kg_stability(self, view1, view2):
        all_items_1 = self.cal_item_embedding_from_kg(view1)  # items * dims
        all_items_2 = self.cal_item_embedding_from_kg(view2)  # items * dims

        # DIFF 性能提升 Cui计算的 u_emb i_emb 经过LightGCN
        if world.hyper_KGCL_my_ablated_model == 1:
            dropout = False
        else:
            dropout = True
        all_users_1, all_items_1 = self.lightGCN(self.embedding_user.weight, all_items_1, self.Graph, dropout=dropout)
        all_users_2, all_items_2 = self.lightGCN(self.embedding_user.weight, all_items_2, self.Graph, dropout=dropout)

        # DIFF 性能提升 计算Cui
        user1_emb = all_users_1[self.ui_dataset.trainUser]  # inters * dims
        user2_emb = all_users_2[self.ui_dataset.trainUser]  # inters * dims
        item1_emb = all_items_1[self.ui_dataset.trainItem]  # inters * dims
        item2_emb = all_items_2[self.ui_dataset.trainItem]  # inters * dims

        self.u1 = all_users_1
        self.u2 = all_users_2
        self.i1 = all_items_1
        self.i2 = all_items_2

        if world.hyper_KGCL_my_ablated_model == 2:
            inter_1 = user1_emb
            inter_2 = user2_emb
        elif world.hyper_KGCL_my_ablated_model == 3:
            inter_1 = item1_emb
            inter_2 = item2_emb
        elif world.hyper_KGCL_my_ablated_model == 4:
            inter_1 = torch.cat((user1_emb, item1_emb), dim=1)  # inters * dims*2
            inter_2 = torch.cat((user2_emb, item2_emb), dim=1)  # inters * dims*2
        else:
            inter_1 = torch.mul(user1_emb, item1_emb)
            inter_2 = torch.mul(user2_emb, item2_emb)

        sim = F.cosine_similarity(inter_1, inter_2)  # inters
        return sim

    def get_kg_views(self):
        view1 = utils.drop_edge_random(self.kg_dict, world.hyper_KGCL_kg_p_drop, self.num_entities)
        view2 = utils.drop_edge_random(self.kg_dict, world.hyper_KGCL_kg_p_drop, self.num_entities)
        return view1, view2

    def cal_item_embedding_from_kg(self, kg: dict):
        item_embs = self.embedding_item(torch.LongTensor(list(kg.keys())).to(world.device))  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.embedding_entity(item_entities.long())  # item_num, entity_num_each, emb_dim
        relation_embs = torch.mul(self.embedding_relation.weight, self.W_R)[item_relations]
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def prepare_each_epoch(self):
        kgv1, kgv2 = self.get_kg_views()
        if world.hyper_KGCL_my_ablated_model == 1:
            kgv2 = kgv1.copy()
        stability = self.item_kg_stability(kgv1, kgv2).to(world.device)
        uiv1 = self.get_ui_views_weighted(stability, 1)
        uiv2 = self.get_ui_views_weighted(stability, 1)
        self.contrast_views = {"kgv1": kgv1, "kgv2": kgv2, "uiv1": uiv1, "uiv2": uiv2}

    def calculate_embedding_graph(self, ui_graph, kg_graph):
        return self.lightGCN(self.embedding_user.weight, self.cal_item_embedding_from_kg(kg_graph), ui_graph)

    def calculate_embedding(self):
        # DIFF 性能提升 在BPR和test中不使用entity
        return self.lightGCN(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        # loss[Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[Loss.BPR.value] = losses.loss_BPR_weighted(all_users, all_items,
                                                        self.u1, self.u2, self.i1, self.i2,
                                                        users, pos, neg)
        # loss[Loss.BPR.value] += losses.loss_SSM_origin(all_users, all_items, users, pos)
        loss[Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        users_v1, items_v1 = self.calculate_embedding_graph(self.contrast_views["uiv1"], self.contrast_views["kgv1"])
        users_v2, items_v2 = self.calculate_embedding_graph(self.contrast_views["uiv2"], self.contrast_views["kgv2"])
        loss[Loss.SSL.value] = losses.loss_info_nce(users_v1, users_v2, users)
        loss[Loss.SSL.value] += losses.loss_info_nce(items_v1, items_v2, pos)
        return loss
