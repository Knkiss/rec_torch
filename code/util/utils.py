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


def convert_sp_mat_to_sp_tensor(x):
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.train_batch_size)
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
    if world.mail_comment:
        content += '\n' + world.mail_comment
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
