from enum import Enum

import numpy as np


class Metrics(Enum):
    Recall = "Recall"
    Precision = "Precision"
    NDCG = "NDCG"


def RecallPrecision_topk(test_data, result, k):
    """
    test_data : 原测试数据
    result : 测试结果
    k : top-k
    """
    right_pred = result[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {Metrics.Recall.value: recall, Metrics.Precision.value: precis}


def NDCG_topK(test_data, result, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
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
