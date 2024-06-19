from enum import Enum

import torch
from sklearn.cluster import KMeans
from torch_scatter import scatter_mean

from train import utils
import world
import torch.nn.functional as F


class Loss(Enum):
    BPR = 'BPR'
    SSL = 'SSL'
    TransE = 'TransE'
    Regulation = 'reg'
    MAE = 'MAE'


def loss_BPR(all_users, all_items, users, pos, neg):
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))  # mean or sum
    return loss


def loss_BPR_weighted(all_users, all_items, u1, u2, i1, i2, users, pos, neg):
    # 使用知识个性化权重的BPR
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    scores = -(pos_scores - neg_scores)

    u1_emb = u1[users.long()]
    u2_emb = u2[users.long()]
    i1_emb = i1[pos.long()]
    i2_emb = i2[pos.long()]

    inter_1 = torch.mul(u1_emb, i1_emb)
    inter_2 = torch.mul(u2_emb, i2_emb)
    sim = F.cosine_similarity(inter_1, inter_2)  # inters
    sim = (sim - sim.min()) / (sim.max() - sim.min())
    # sim = torch.exp(sim)
    # item_stabilities = t
    # orch.exp(item_stabilities)

    logits = sim.detach()

    loss = torch.sum(torch.nn.functional.softplus(scores * logits))
    return loss


def loss_SSM_origin(all_users, all_items, users, pos):
    batch_user_emb = F.normalize(all_users[users.long()], dim=1)
    batch_item_emb = F.normalize(all_items[pos.long()], dim=1)
    pos_score = torch.sum(torch.multiply(batch_user_emb, batch_item_emb), dim=1, keepdim=True)
    ttl_score = torch.matmul(batch_user_emb, batch_item_emb.T) * world.hyper_SSM_Margin
    logits = ttl_score - pos_score
    clogits = torch.logsumexp(logits / world.hyper_SSM_Loss_temp, dim=1)
    loss = torch.sum(clogits)
    return loss * world.hyper_SSM_Regulation


def loss_New_1_1(all_users, all_items, users, pos, neg):
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]

    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)

    neg_scores = torch.matmul(users_emb, neg_emb.T)
    neg_scores = torch.mean(neg_scores, dim=1)

    loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
    return loss


def loss_New_1_2(all_users, all_items, users, pos, neg):
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]

    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)

    neg_scores = torch.matmul(users_emb, neg_emb.T)
    neg_scores = torch.exp(neg_scores)
    neg_scores = torch.mean(neg_scores, dim=1)
    neg_scores = torch.log(neg_scores)

    loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
    return loss


def loss_New_1_3(all_users, all_items, users, pos, neg):
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]

    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)

    neg_scores_1 = torch.mul(users_emb, neg_emb).sum(dim=1)

    neg_scores = torch.matmul(users_emb, pos_emb.T)
    neg_scores = torch.exp(neg_scores)
    neg_scores = torch.mean(neg_scores, dim=1)
    neg_scores = torch.log(neg_scores)

    loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores_1 - neg_scores)))
    return loss


def loss_New_2_1(all_users, all_items, users, pos, neg):
    batch_user_emb = F.normalize(all_users[users.long()], dim=1)
    batch_item_emb = F.normalize(all_items[pos.long()], dim=1)
    batch_item_emb_neg = F.normalize(all_items[neg.long()], dim=1)

    pos = torch.mul(batch_user_emb, batch_item_emb).sum(dim=1)
    neg = torch.mul(batch_user_emb, batch_item_emb_neg).sum(dim=1)
    pos_score = torch.nn.functional.softplus(pos - neg).unsqueeze(dim=1)

    ttl_score_neg_all = torch.matmul(batch_user_emb, batch_item_emb.T) * world.hyper_SSM_Margin
    ttl_score_neg = torch.matmul(batch_user_emb, batch_item_emb_neg.T) * world.hyper_SSM_Margin
    ttl_score = ttl_score_neg_all + ttl_score_neg

    logits = ttl_score - pos_score
    clogits = torch.logsumexp(logits / world.hyper_SSM_Loss_temp, dim=1)
    loss = torch.sum(clogits)
    return loss


def loss_New_2_2(all_users, all_items, users, pos, neg):
    batch_user_emb = F.normalize(all_users[users.long()], dim=1)
    batch_item_emb = F.normalize(all_items[pos.long()], dim=1)
    batch_neg_emb = F.normalize(all_items[neg.long()], dim=1)

    pos_score = torch.sum(torch.multiply(batch_user_emb, batch_item_emb), dim=1)
    pos_score = torch.exp(pos_score / world.hyper_SSM_Loss_temp)

    neg_score = torch.sum(torch.multiply(batch_user_emb, batch_neg_emb), dim=1)
    neg_score = torch.exp(neg_score / world.hyper_SSM_Loss_temp)

    ttl_score = torch.matmul(batch_user_emb, batch_item_emb.T)
    ttl_score = torch.sum(torch.exp(ttl_score / world.hyper_SSM_Loss_temp), dim=1)

    clogits = -(torch.log(pos_score) - torch.log(
        ttl_score * world.hyper_test_ratio + neg_score * world.hyper_test_ratio_2))
    loss = torch.sum(clogits)
    return loss


def loss_New_2_3(all_users, all_items, users, pos, neg):
    batch_user_emb = F.normalize(all_users[users.long()], dim=1)
    batch_item_emb = F.normalize(all_items[pos.long()], dim=1)
    batch_neg_emb = F.normalize(all_items[neg.long()], dim=1)

    pos_score = torch.sum(torch.multiply(batch_user_emb, batch_item_emb), dim=1)
    pos_score = torch.exp(pos_score / world.hyper_SSM_Loss_temp)

    neg_score = torch.sum(torch.multiply(batch_user_emb, batch_neg_emb), dim=1)
    neg_score = torch.exp(neg_score / world.hyper_SSM_Loss_temp)

    ttl_score = torch.matmul(batch_user_emb, batch_item_emb.T)
    ttl_score = torch.sum(torch.exp(ttl_score / world.hyper_SSM_Loss_temp), dim=1)

    clogits = -(torch.log(pos_score) - torch.log(ttl_score * neg_score))
    loss = torch.sum(clogits)
    return loss


def loss_regulation(all_users_origin, all_items_origin, users, pos, neg):
    if isinstance(all_users_origin, torch.Tensor):
        userEmb0 = all_users_origin[users.long()]
        posEmb0 = all_items_origin[pos.long()]
        negEmb0 = all_items_origin[neg.long()]
    else:
        userEmb0 = all_users_origin(users.long())
        posEmb0 = all_items_origin(pos.long())
        negEmb0 = all_items_origin(neg.long())
    loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
    return loss * world.hyper_decay


def loss_info_nce(node_v1, node_v2, batch):
    # KGCL文章代码中的对比Loss计算形式
    z1 = node_v1[batch]
    z2 = node_v2[batch]
    z_all = node_v2

    def f(x):
        return torch.exp(x / world.hyper_ssl_temp)

    all_sim = f(utils.sim(z1, z_all))
    positive_pairs = f(utils.sim(z1, z2))
    negative_pairs = torch.sum(all_sim, 1)
    loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
    return loss * world.hyper_ssl_reg


def loss_SGL(node_v1, node_v2, batch):
    # SGL文章代码中的对比Loss计算形式
    emb1 = node_v1[batch]
    emb2 = node_v2[batch]

    normalize_emb1 = F.normalize(emb1, 1)
    normalize_emb2 = F.normalize(emb2, 1)
    normalize_all_emb2 = F.normalize(node_v2, dim=1)

    pos_score = torch.sum(torch.mul(normalize_emb1, normalize_emb2), dim=1)
    ttl_score = torch.matmul(normalize_emb1, normalize_all_emb2.T)

    pos_score = torch.exp(pos_score / world.hyper_ssl_temp)
    ttl_score = torch.sum(torch.exp(ttl_score / world.hyper_ssl_temp), dim=1)

    loss = -torch.sum(torch.log(pos_score / ttl_score))
    return loss * world.hyper_ssl_reg


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
        return torch.mean(torch.div(torch.sum(torch.pow(x, 2), dim=1, keepdim=False), 2.))

    l2_loss = L2_loss_mean(h_embed) + L2_loss_mean(r_embed) + L2_loss_mean(pos_t_embed) + L2_loss_mean(neg_t_embed)
    loss = kg_loss + 1e-3 * l2_loss
    return loss


def loss_kd_cluster_ii_graph(from_emb, to_emb):
    cluster_num = world.hyper_WORK2_cluster_num
    source_emb_np = from_emb.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init="auto")
    kmeans.fit(source_emb_np)
    idx = torch.LongTensor(kmeans.labels_).to(world.device)

    from_cluster = scatter_mean(src=from_emb, index=idx, dim_size=cluster_num, dim=0)
    to_cluster = scatter_mean(src=to_emb, index=idx, dim_size=cluster_num, dim=0)

    # 构造kmeans后的ii中心矩阵
    ckg_constructed_graph = torch.cosine_similarity(from_cluster.unsqueeze(dim=0), from_cluster.unsqueeze(dim=1), 2)
    ui_constructed_graph = torch.cosine_similarity(to_cluster.unsqueeze(dim=0), to_cluster.unsqueeze(dim=1), 2)
    kd_loss = torch.sum((ckg_constructed_graph - ui_constructed_graph) ** 2) * world.hyper_KD_regulation
    return kd_loss
