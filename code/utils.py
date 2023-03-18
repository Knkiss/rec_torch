# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：utils.py
@Author  ：Knkiss
@Date    ：2023/2/14 12:14 
"""
import smtplib
from email.mime.text import MIMEText
from random import random

import numpy as np
import torch
import torch.nn.functional as F

import world
from main import Metrics


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def loss_BPR(all_users, all_items, users, pos, neg):
    users_emb = all_users[users.long()]
    pos_emb = all_items[pos.long()]
    neg_emb = all_items[neg.long()]
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))  # mean or sum
    return loss


def loss_regulation(all_users_origin, all_items_origin, users, pos, neg):
    userEmb0 = all_users_origin(users.long())
    posEmb0 = all_items_origin(pos.long())
    negEmb0 = all_items_origin(neg.long())
    loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
    return loss * world.decay


def loss_info_nce(node_v1, node_v2, batch):
    z1 = node_v1[batch]
    z2 = node_v2[batch]
    z_all = node_v2

    def f(x):
        return torch.exp(x / world.ssl_temp)

    all_sim = f(sim(z1, z_all))
    positive_pairs = f(sim(z1, z2))
    negative_pairs = torch.sum(all_sim, 1)
    loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
    return loss * world.ssl_reg


def loss_SGL(node_v1, node_v2, batch):
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

    l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
        neg_t_embed)
    loss = kg_loss + 1e-3 * l2_loss
    return loss


def sim(z1: torch.Tensor, z2: torch.Tensor):
    if z1.size()[0] == z2.size()[0]:
        return F.cosine_similarity(z1, z2)
    else:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


def dropout_x(x, keep_prob):
    size = x.size()
    index = x.indices().t()
    values = x.values()
    random_index = torch.rand(len(values)) + keep_prob
    random_index = random_index.int().bool()
    index = index[random_index]
    values = values[random_index] / keep_prob
    g = torch.sparse.FloatTensor(index.t(), values, size)
    return g


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


def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def convert_sp_mat_to_sp_tensor(x):
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(int(timer.time() - self.start))


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {Metrics.Recall.value: recall, Metrics.Precision.value: precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['train_batch_size'])
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def mail_on_stop(results):
    if not world.mail_on_stop_enable:
        return

    content = '模型: ' + str(world.model) + '\n数据集: ' + str(world.dataset) + '\n结果: ' + str(results)
    if world.comment:
        content += '\n' + world.comment
    message = MIMEText(content, 'plain', 'utf-8')
    message['Subject'] = '服务器代码运行完毕'
    message['From'] = world.mail_sender
    message['To'] = world.mail_receivers[0]

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(world.mail_host, 25)
        smtpObj.login(world.mail_user, world.mail_pass)
        smtpObj.sendmail(world.mail_sender, world.mail_receivers, message.as_string())
        smtpObj.quit()
        print('发送提醒邮件至' + world.mail_receivers[0])
    except smtplib.SMTPException as e:
        print('发送邮件错误：', e)


if __name__ == '__main__':
    mail_on_stop('1')
