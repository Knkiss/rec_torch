from enum import Enum

import numpy as np
import torch
import world
from train import utils
from model import abstract_model


class Metrics(Enum):
    Recall = "Recall"
    Precision = "Precision"
    NDCG = "NDCG"
    MRR = "MRR"


def Recall_topK(test_data, result, k):
    right_pred = result[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    return np.sum(right_pred / recall_n)


def Precision_topK(result, k):
    right_pred = result[:, :k].sum(1)
    return np.sum(right_pred) / k


def NDCG_topK(test_data, result, k):
    assert len(result) == len(test_data)
    pred_data = result[:, :k]
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


def MRR_topK(result, k):
    pred_data = result[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


# def AUC(all_item_scores, dataset, test_data):
#     dataset : UIDataset
#     r_all = np.zeros((dataset.m_items, ))
#     r_all[test_data] = 1
#     r = r_all[all_item_scores >= 0]
#     test_item_scores = all_item_scores[all_item_scores >= 0]
#     return roc_auc_score(r, test_item_scores)

def MAD_embedding(model: abstract_model):
    all_users, all_items = model.calculate_embedding()
    all_e = torch.concat([all_users, all_items], dim=0)
    eq2 = []
    graph = model.Graph.clone()
    graph.values().fill_(1)
    graph = graph.to_dense()
    eq2_neighbour = []
    eq2_remote = []
    for batch in utils.minibatch(list(range(all_e.shape[0])), batch_size=world.test_u_batch_size // 4):
        bat_e = all_e[batch]
        eq1 = 1 - torch.clamp(torch.cosine_similarity(all_e[None, :, :], bat_e[:, None, :], dim=-1), -1.0, 1.0)
        eq1_eps = torch.round(torch.abs(eq1) * 1e6) / 1e6  # 计算会产生误差，消除1e6以上的误差
        eq2.append((torch.sum(eq1_eps, dim=1) + 1e-8) / (torch.where(eq1_eps > 0, 1, 0).sum(dim=1) + 1e-8))

        eq1_neighbour = eq1_eps * graph[batch]
        eq1_remote = eq1_eps - eq1_neighbour
        eq2_neighbour.append((torch.sum(eq1_neighbour, dim=1) + 1e-8) / (torch.where(eq1_neighbour > 0, 1, 0).sum(dim=1) + 1e-8))
        eq2_remote.append((torch.sum(eq1_remote, dim=1) + 1e-8) / (torch.where(eq1_remote > 0, 1, 0).sum(dim=1) + 1e-8))

    eq2 = torch.concat(eq2, dim=0)
    eq2_neighbour = torch.concat(eq2_neighbour, dim=0)
    eq2_remote = torch.concat(eq2_remote, dim=0)
    MAD = torch.sum(eq2, dim=0) / torch.where(eq2 > 0, 1, 0).sum(dim=0)
    MAD_neighbour = torch.sum(eq2_neighbour, dim=0) / torch.where(eq2_neighbour > 0, 1, 0).sum(dim=0)
    MAD_remote = torch.sum(eq2_remote, dim=0) / torch.where(eq2_remote > 0, 1, 0).sum(dim=0)
    MADGap = MAD_remote - MAD_neighbour
    return MAD, MADGap
