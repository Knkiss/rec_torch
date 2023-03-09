# -*- coding: utf-8 -*-

from cppimport import imp
from numpy import negative, positive
from torch.utils.data import DataLoader
from torch_sparse.tensor import to
from tqdm import tqdm

import utils
from model import KGCL
from random import random, sample
from shutil import make_archive
import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected
from utils import randint_choice
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import world

"""
graph shape:
[    0,     0,     0,  ..., 69714, 69715, 69715],
[    0, 31668, 31669,  ..., 69714, 31666, 69715]

values=tensor([0.0526, 0.0096, 0.0662,  ..., 0.5000, 0.1443, 0.5000])
"""


def drop_edge_random(item2entities, p_drop, padding):
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if random() > p_drop:
                new_es.append(e)
            else:
                new_es.append(padding)
        res[item] = torch.IntTensor(new_es).to(world.device)
    return res


def drop_edge_weighted(edge_index, edge_weights, p: float = 0.3, threshold: float = 0.7):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


class Contrast(nn.Module):
    def __init__(self, gcn_model, tau=world.kgc_temp):
        super(Contrast, self).__init__()
        self.gcn_model: KGCL = gcn_model
        self.tau = tau

    def BPR_train_contrast(self, contrast_views, epoch, opt, w=None):
        self.gcn_model.train()
        batch_size = world.config['bpr_batch_size']
        dataloader = DataLoader(self.gcn_model.ui_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                num_workers=0)

        total_batch = len(dataloader)
        aver_loss = 0.
        aver_loss_main = 0.
        aver_loss_ssl = 0.

        uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
        kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]

        for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=False):
            batch_users = train_data[0].long().to(world.device)
            batch_pos = train_data[1].long().to(world.device)
            batch_neg = train_data[2].long().to(world.device)

            l_main = utils.computeBPR(self.gcn_model, batch_users, batch_pos, batch_neg)
            l_ssl = list()

            usersv1_ro, itemsv1_ro = self.gcn_model.lightGCN_drop(uiv1, kgv1)
            usersv2_ro, itemsv2_ro = self.gcn_model.lightGCN_drop(uiv2, kgv2)

            items_uiv1 = itemsv1_ro[batch_pos]
            items_uiv2 = itemsv2_ro[batch_pos]
            l_item = self.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro)

            users_uiv1 = usersv1_ro[batch_users]
            users_uiv2 = usersv2_ro[batch_users]
            l_user = self.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro)

            l_ssl.extend([l_user * world.ssl_reg, l_item * world.ssl_reg])

            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main + l_ssl
            aver_loss_ssl += l_ssl.cpu().item()

            opt.zero_grad()
            l_all.backward()
            opt.step()

            aver_loss_main += l_main.cpu().item()
            aver_loss += l_all.cpu().item()

        aver_loss = aver_loss / (total_batch * batch_size)
        aver_loss_main = aver_loss_main / (total_batch * batch_size)
        aver_loss_ssl = aver_loss_ssl / (total_batch * batch_size)

        w.add_scalar(f'Loss/BPR', aver_loss_main, epoch)
        w.add_scalar(f'Loss/SSL', aver_loss_ssl, epoch)

        return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}"

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def info_nce_loss_overall(self, z1, z2, z_all):
        f = lambda x: torch.exp(x / self.tau)
        # batch_size
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.pair_sim(z1, z1))
        between_sim = f(self.pair_sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size()
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = False, batch_size: int = 0):
        # h1 = self.projection(z1)
        # h2 = self.projection(z2)
        h1 = z1
        h2 = z2

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def get_ui_views_pgrace(self):
        n_users = self.gcn_model.num_users
        n_items = self.gcn_model.num_items
        graph = self.gcn_model.dataset.UserItemNet
        # user_num
        user_deg = np.squeeze(np.asarray(graph.sum(1)))
        # item_num
        item_deg = np.squeeze(np.asarray(graph.sum(0)))
        s_col = torch.log(torch.from_numpy(item_deg))
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
        # p control
        edge_weights = weights / weights.mean() * 0.9
        edge_weights = edge_weights.where(edge_weights < 1, torch.ones_like(edge_weights) * 1)
        # item_num
        drop_mask = torch.bernoulli(1. - edge_weights).to(torch.bool).cpu().tolist()

        n_nodes = n_users + n_items
        # [interaction_num]
        item_np = self.gcn_model.dataset.trainItem
        keep_idx = list()
        for i, j in enumerate(item_np.tolist()):
            if not drop_mask[j]:
                keep_idx.append(i)
            else:
                r = random()
                if r < 0.6:
                    keep_idx.append(i)
        print(f"finally keep ratio: {len(keep_idx) / len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        item_np = item_np[keep_idx]
        user_np = self.gcn_model.dataset.trainUser[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
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

    def item_kg_stability(self, view1, view2):
        kgv1_ro = self.gcn_model.cal_item_embedding_from_kg(view1)  # items * dims
        kgv2_ro = self.gcn_model.cal_item_embedding_from_kg(view2)  # items * dims
        if world.user_item_preference:
            userList = torch.LongTensor(self.gcn_model.ui_dataset.trainUser).to(world.device)
            user1_emb = self.gcn_model.embedding_user(userList)  # inters * dims
            user2_emb = self.gcn_model.embedding_user(userList)  # inters * dims
            item1_emb = kgv1_ro[self.gcn_model.ui_dataset.trainItem]  # inters * dim
            item2_emb = kgv2_ro[self.gcn_model.ui_dataset.trainItem]  # inters * dim
            ui1_emb = torch.cat((user1_emb, item1_emb), dim=1)  # inters * dim*2
            ui2_emb = torch.cat((user2_emb, item2_emb), dim=1)  # inters * dim*2
            sim = F.cosine_similarity(ui1_emb, ui2_emb)  # inters 交互数量
        else:
            sim = self.sim(kgv1_ro, kgv2_ro)  # items 物品数量
        return sim

    def ui_drop_weighted(self, item_mask):
        item_mask = item_mask.tolist()
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        item_np = self.gcn_model.ui_dataset.trainItem
        keep_idx = list()
        if world.user_item_preference:
            for i, j in enumerate(item_mask):
                if j:
                    keep_idx.append(i)
        else:
            for i, j in enumerate(item_np.tolist()):
                if item_mask[j]:
                    keep_idx.append(i)


        print(f"finally keep ratio: {len(keep_idx) / len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        user_np = self.gcn_model.ui_dataset.trainUser[keep_idx]
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
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

    def get_views(self):
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
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
            kg, _ = self.gcn_model.kg_dataset.get_kg_dict_random(self.gcn_model.num_items)
        else:
            kg = self.gcn_model.kg_dict
        view1 = drop_edge_random(kg, world.kg_p_drop, self.gcn_model.num_entities)
        view2 = drop_edge_random(kg, world.kg_p_drop, self.gcn_model.num_entities)
        return view1, view2

    def get_ui_views_weighted(self, item_stabilities, stab_weight):
        graph = self.gcn_model.Graph
        n_users = self.gcn_model.num_users

        # generate mask
        item_degrees = degree(graph.indices()[0])[n_users:].tolist()
        deg_col = torch.FloatTensor(item_degrees).to(world.device)
        s_col = torch.log(deg_col)
        # degree normalization
        # deg probability of keep
        degree_weights = (s_col - s_col.min()) / (s_col.max() - s_col.min())
        degree_weights = degree_weights.where(degree_weights > 0.3, torch.ones_like(degree_weights) * 0.3)  # p_tau

        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        kg_weights = (item_stabilities - item_stabilities.min()) / (item_stabilities.max() - item_stabilities.min())
        kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = (1 - world.ui_p_drop) / torch.mean(stab_weight * kg_weights) * (stab_weight * kg_weights)
        weights = weights.where(weights < 0.95, torch.ones_like(weights) * 0.95)

        item_mask = torch.bernoulli(weights).to(torch.bool)
        print(f"keep ratio: {item_mask.sum() / item_mask.size()[0]:.2f}")
        # drop
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        return g_weighted
