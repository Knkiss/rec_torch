from enum import Enum

import numpy as np


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
