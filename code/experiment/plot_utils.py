import numpy as np


def metric_precision_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def metric_map(r, cut):
    r = np.asarray(r)
    out = [metric_precision_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / float(min(cut, np.sum(r)))


def metric_dcg_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def metric_ndcg_k(r, k, method=1):
    dcg_max = metric_dcg_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return metric_dcg_k(r, k, method) / dcg_max


def metric_recall_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def metric_hit_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


def get_performance_user(x, u, test_set, Ks):
    ranks = x
    user_pos_test = test_set[u]
    r = []
    for i in ranks:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    precision, recall, ndcg, hit_ratio, ap = [], [], [], [], []
    for K in Ks:
        precision.append(metric_precision_k(r, K))
        recall.append(metric_recall_k(r, K, len(user_pos_test)))
        ndcg.append(metric_ndcg_k(r, K))
        hit_ratio.append(metric_hit_k(r, K))
        ap.append(metric_map(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'map': np.array(ap)}
