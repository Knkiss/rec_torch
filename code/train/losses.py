from enum import Enum

import torch

from util import utils
import world
import torch.nn.functional as F


class Loss(Enum):
    BPR = 'BPR'
    SSL = 'SSL'
    TransE = 'TransE'
    Regulation = 'reg'


def loss_BPR(all_users, all_items, users, pos, neg):
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))  # mean or sum
    return loss


def loss_SSM(all_users, all_items, users, pos):
    SSM_sampled_number = pos.shape[0]

    user_emb = all_users[users.long()]  # 【N,d】
    pos_emb = all_items[pos.long()]  # 【N,d】
    positive_pairs = torch.exp(F.cosine_similarity(user_emb, pos_emb, dim=1))  # 【N】

    user_emb = user_emb.unsqueeze(1).repeat(1, SSM_sampled_number, 1)  # 【N，sample, d】
    neg_emb = all_items[pos.long()][:SSM_sampled_number, :].unsqueeze(0).repeat(users.shape[0], 1, 1)  # 【N, sample, d】
    all_sim = torch.exp(F.cosine_similarity(user_emb, neg_emb, dim=2))  # 【N, sample】
    negative_pairs = torch.sum(all_sim, 1)  # 【N】

    loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
    return loss


def loss_regulation(all_users_origin, all_items_origin, users, pos, neg):
    userEmb0 = all_users_origin(users.long())
    posEmb0 = all_items_origin(pos.long())
    negEmb0 = all_items_origin(neg.long())
    loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
    return loss * world.decay


def loss_info_nce(node_v1, node_v2, batch):
    # KGCL文章代码中的对比Loss计算形式
    z1 = node_v1[batch]
    z2 = node_v2[batch]
    z_all = node_v2

    def f(x):
        return torch.exp(x / world.ssl_temp)

    all_sim = f(utils.sim(z1, z_all))
    positive_pairs = f(utils.sim(z1, z2))
    negative_pairs = torch.sum(all_sim, 1)
    loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
    return loss * world.ssl_reg


def loss_SGL(node_v1, node_v2, batch):
    # SGL文章代码中的对比Loss计算形式
    emb1 = node_v1[batch]
    emb2 = node_v2[batch]

    normalize_emb1 = F.normalize(emb1, 1)
    normalize_emb2 = F.normalize(emb2, 1)
    normalize_all_emb2 = F.normalize(node_v2, dim=1)

    pos_score = torch.sum(torch.mul(normalize_emb1, normalize_emb2), dim=1)
    ttl_score = torch.matmul(normalize_emb1, normalize_all_emb2.T)

    pos_score = torch.exp(pos_score / world.ssl_temp)
    ttl_score = torch.sum(torch.exp(ttl_score / world.ssl_temp), dim=1)

    loss = -torch.sum(torch.log(pos_score / ttl_score))
    return loss * world.ssl_reg


def loss_transE(head, tail, relation, h, r, pos_t, neg_t):
    r_embed = relation(r)
    h_embed = head(h)
    pos_t_embed = tail(pos_t)
    neg_t_embed = tail(neg_t)

    pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)
    neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)

    kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
    kg_loss = torch.mean(kg_loss)

    def L2_loss_mean(x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

    l2_loss = L2_loss_mean(h_embed) + L2_loss_mean(r_embed) + L2_loss_mean(pos_t_embed) + L2_loss_mean(neg_t_embed)
    loss = kg_loss + 1e-3 * l2_loss
    return loss
