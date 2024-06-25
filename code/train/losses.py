import random
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


# mode = 1, 2
def loss_kd_ii_graph_batch(from_item_emb, to_item_emb, batch_item, batch_item_neg=None):
    if batch_item_neg is None:
        t = from_item_emb[batch_item.long()]
        s = to_item_emb[batch_item.long()]
    else:
        t = torch.concat([from_item_emb[batch_item.long()], from_item_emb[batch_item_neg.long()]], dim=0)
        s = torch.concat([to_item_emb[batch_item.long()], to_item_emb[batch_item_neg.long()]], dim=0)

    t_dist = torch.cosine_similarity(t.unsqueeze(dim=0), t.unsqueeze(dim=1), dim=2)
    s_dist = torch.cosine_similarity(s.unsqueeze(dim=0), s.unsqueeze(dim=1), dim=2)

    kd_loss = torch.sum((t_dist - s_dist) ** 2) * world.hyper_KD_regulation
    return kd_loss


# mode = 3, 4
def loss_kd_cluster_ii_graph_batch(from_item_emb, to_item_emb, batch_item=None, gpu=True):
    if batch_item is not None:
        from_item_emb = from_item_emb[batch_item.long()]
        to_item_emb = to_item_emb[batch_item.long()]
    cluster_num = world.hyper_WORK2_cluster_num

    if gpu:
        _, idx = utils.kmeans(from_item_emb, world.hyper_WORK2_cluster_num)
    else:
        source_emb_np = from_item_emb.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=cluster_num, random_state=random.randint(0, 10000), n_init="auto")
        kmeans.fit(source_emb_np)
        idx = torch.LongTensor(kmeans.labels_).to(world.device)

    from_cluster = scatter_mean(src=from_item_emb, index=idx, dim_size=cluster_num, dim=0)
    to_cluster = scatter_mean(src=to_item_emb, index=idx, dim_size=cluster_num, dim=0)

    # 构造kmeans后的ii中心矩阵
    ckg_constructed_graph = torch.cosine_similarity(from_cluster.unsqueeze(dim=0), from_cluster.unsqueeze(dim=1), 2)
    ui_constructed_graph = torch.cosine_similarity(to_cluster.unsqueeze(dim=0), to_cluster.unsqueeze(dim=1), 2)
    kd_loss = torch.sum((ckg_constructed_graph - ui_constructed_graph) ** 2) * world.hyper_KD_regulation
    return kd_loss


# mode = 5
def loss_kd_A_graph_batch(from_user_emb, to_user_emb, from_item_emb, to_item_emb, batch_user, batch_item):
    t = torch.cat([from_user_emb[batch_user.long()], from_item_emb[batch_item.long()]], 0)
    s = torch.cat([to_user_emb[batch_user.long()], to_item_emb[batch_item.long()]], 0)

    t_dist = torch.cosine_similarity(t.unsqueeze(dim=0), t.unsqueeze(dim=1), dim=2)
    s_dist = torch.cosine_similarity(s.unsqueeze(dim=0), s.unsqueeze(dim=1), dim=2)

    kd_loss = torch.sum((t_dist - s_dist) ** 2) * world.hyper_KD_regulation
    return kd_loss


# mode = 6, 7
def loss_kd_mlp_ii_graph_batch(group_mlp, from_item_emb, to_item_emb, batch_item=None, eps=1e-10, tau=0.0001):
    if batch_item is not None:
        from_item_emb = from_item_emb[batch_item.long()]
        to_item_emb = to_item_emb[batch_item.long()]

    group_emb = group_mlp(from_item_emb)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(group_emb) + eps) + eps)
    gumbel_logits = (group_emb + gumbel_noise) / tau  # tau=0.001即可实现0和1的softmax，因此设置为0.0001
    group_labels = F.softmax(gumbel_logits, dim=-1)

    from_cluster = torch.mm(group_labels.T, from_item_emb)
    to_cluster = torch.mm(group_labels.T, to_item_emb)

    # 构造mlp分组后的ii中心矩阵
    ckg_constructed_graph = torch.cosine_similarity(from_cluster.unsqueeze(dim=0), from_cluster.unsqueeze(dim=1), 2)
    ui_constructed_graph = torch.cosine_similarity(to_cluster.unsqueeze(dim=0), to_cluster.unsqueeze(dim=1), 2)
    kd_loss = torch.sum((ckg_constructed_graph - ui_constructed_graph) ** 2) * world.hyper_KD_regulation
    return kd_loss


# mode = 8, 9 存疑
def loss_bpr_mlp_ui_graph_batch(group_mlp, from_item_emb, to_item_emb, from_user_emb, to_user_emb, users, pos, neg,
                                eps=1e-10, tau=0.0001, form='BPR', gumbel_softmax=2):
    from_item_emb = F.dropout(from_item_emb, 0.05)
    group_emb = group_mlp(from_item_emb)

    if gumbel_softmax == 2:
        p = F.softmax(group_emb, dim=1)
        g = - torch.log(-torch.log(torch.rand_like(group_emb) + 1e-25))
        gumbel_logits = (torch.log(p) + g) / tau
    else:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(group_emb) + eps) + eps)
        gumbel_logits = (group_emb + gumbel_noise) / tau  # tau=0.001即可实现0和1的softmax，因此设置为0.0001
    group_labels = F.softmax(gumbel_logits, dim=-1)
    # 从CKG提供的item group labels

    pos_batch_idx = torch.argmax(group_labels[pos], dim=1)
    neg_batch_idx = torch.argmax(group_labels[neg], dim=1)

    # 用于对UI的兴趣分布进行优化
    # from_cluster = torch.mm(group_labels.T, from_item_emb)
    to_cluster = torch.mm(group_labels.T, to_item_emb)

    if form == 'BPR':
        users_emb = to_user_emb[users]
        pos_batch_group_emb = to_cluster[pos_batch_idx]
        neg_batch_group_emb = to_cluster[neg_batch_idx]
        pos_scores = torch.mul(users_emb, pos_batch_group_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_batch_group_emb).sum(dim=1)
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores))) * world.hyper_KD_regulation
    elif form == 'InfoNCE':
        normalize_all_emb1 = F.normalize(to_user_emb, 1)
        normalize_all_emb2 = F.normalize(to_cluster, 1)
        normalize_emb1 = normalize_all_emb1[users]
        normalize_emb2 = normalize_all_emb2[pos_batch_idx]
        pos_score = torch.sum(torch.mul(normalize_emb1, normalize_emb2), dim=1)
        ttl_score = torch.matmul(normalize_emb1, normalize_all_emb2.T)
        pos_score = torch.exp(pos_score / world.hyper_ssl_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / world.hyper_ssl_temp), dim=1)
        loss = -torch.sum(torch.log(pos_score / ttl_score)) * world.hyper_KD_regulation
    else:
        raise NotImplementedError("form = [BPR, InfoNCE]")
    return loss
