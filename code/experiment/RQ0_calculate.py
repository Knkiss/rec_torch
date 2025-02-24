# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
from tqdm import tqdm

import world
import plot_utils


def calculate_performance_all(rank_new, Ks, test_set):
    n_users = len(rank_new)
    r = {'precision': np.zeros(len(Ks)),
         'recall': np.zeros(len(Ks)),
         'ndcg': np.zeros(len(Ks)),
         'hit_ratio': np.zeros(len(Ks)),
         'map': np.zeros(len(Ks))}
    for u in range(n_users):
        re = plot_utils.get_performance_user(rank_new[u], u, test_set, Ks)
        r['precision'] += re['precision'] / n_users
        r['recall'] += re['recall'] / n_users
        r['ndcg'] += re['ndcg'] / n_users
        r['hit_ratio'] += re['hit_ratio'] / n_users
        r['map'] += re['map'] / n_users
    return r


def calculate_group_item(rank_new, i_num, group_num, rank, Ks, test_set, dataset, model):
    """
    计算分组性能
    @return: 分组性能
    """
    n_users = len(rank_new)
    group = []
    sorted_id = sorted(range(len(i_num)), key=lambda k: i_num[k], reverse=True)

    i_count = {}
    average_num = sum(i_num) / group_num
    ends = []  # 每个组内的物品数量，后者包含前者
    i = 0
    for g in range(group_num - 1):
        count = 0
        while count < average_num:
            count += i_num[sorted_id[i]]
            i += 1
        ends.append(i)
    ends.append(len(i_num))

    for n in rank:
        if i_count.get(n) is None:
            i_count[n] = 1
        else:
            i_count[n] += 1

    rank_split = [[] for _ in range(group_num)]
    # test_set_new = [[] for _ in range(group_num)]
    for g in range(group_num):
        if g == 0:
            start = 0
        else:
            start = ends[g - 1]
        end = ends[g]
        group.append(sorted_id[start:end])

        for u in range(n_users):
            a = []
            # test_set_new[g].append([val for val in test_set[u] if val in group[g]])
            for val in rank_new[u]:
                a.append(val if val in group[g] else 'a')
            rank_split[g].append(a)

    g_num = []  # 每组的交互数量
    g_class = []  # 每组的交互物品种类数量
    for g in range(len(group)):
        g_num.append(0)
        g_class.append(0)
        for idx in group[g]:
            if idx in i_count.keys():
                g_num[g] += i_count[idx]
                g_class[g] += 1

    result_g = [{'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
                 'ndcg': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks)),
                 'map': np.zeros(len(Ks))} for _ in range(group_num)]
    with tqdm(range(n_users), desc="Item Group Performance Calculate") as t:
        t.set_postfix(dataset=dataset, model=model)
        for u in t:
            for g in range(group_num):
                re = plot_utils.get_performance_user(rank_split[g][u], u, test_set, Ks)
                result_g[g]['precision'] += re['precision'] / n_users
                result_g[g]['recall'] += re['recall'] / n_users
                result_g[g]['ndcg'] += re['ndcg'] / n_users
                result_g[g]['hit_ratio'] += re['hit_ratio'] / n_users
                result_g[g]['map'] += re['map'] / n_users

    g_interactions_limit = []
    for g in range(group_num):
        g_interactions_limit.append(i_num[sorted_id[ends[g] - 1]])

    return [result_g, g_class, g_num, g_interactions_limit]
    # 每组的性能指标、每组的交互物品种类数量、每组的交互数量、横轴标签：每个物品分组的最小交互个数


def calculate_group_user(rank_new, u_num, group_num, rank, Ks, test_set, dataset, model):
    group = []
    sorted_id = sorted(range(len(u_num)), key=lambda k: u_num[k], reverse=True)
    i_count = {}
    average_num = sum(u_num) / group_num
    ends = []

    i = 0
    for g in range(group_num - 1):
        count = 0
        while count < average_num:
            count += u_num[sorted_id[i]]
            i += 1
        ends.append(i)
    ends.append(len(u_num))

    for n in rank:
        if i_count.get(n) is None:
            i_count[n] = 1
        else:
            i_count[n] += 1

    for g in range(group_num):
        if g == 0:
            start = 0
        else:
            start = ends[g - 1]
        end = ends[g]
        group.append(sorted_id[start:end])

    g_num, g_class = [], []
    for g in range(len(group)):
        g_num.append(0)
        g_class.append(0)
        for idx in group[g]:
            if idx in i_count.keys():
                g_num[g] += i_count[idx]
                g_class[g] += 1
    result_g = [{'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
                 'ndcg': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks)),
                 'map': np.zeros(len(Ks))} for _ in range(group_num)]

    with tqdm(range(group_num), desc="User Group Performance Calculate") as t:
        t.set_postfix(dataset=dataset, model=model)
        for g in t:
            for u in group[g]:
                re = plot_utils.get_performance_user(rank_new[u], u, test_set, Ks)
                result_g[g]['precision'] += re['precision'] / len(group[g])
                result_g[g]['recall'] += re['recall'] / len(group[g])
                result_g[g]['ndcg'] += re['ndcg'] / len(group[g])
                result_g[g]['hit_ratio'] += re['hit_ratio'] / len(group[g])
                result_g[g]['map'] += re['map'] / len(group[g])

    g_interactions_limit = []
    for g in range(group_num):
        g_interactions_limit.append(u_num[sorted_id[ends[g] - 1]])

    return [result_g, g_class, g_num, g_interactions_limit]
    # 每组的性能指标、每组的交互物品种类数量、每组的交互数量、横轴标签：每个用户分组的最小交互个数


def record_result(r, r_ig, r_ug, dataset, model, debug):
    record_dir = world.PATH_RECORD
    record_file = join(record_dir, dataset + '_' + model + '.npy')

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    if not os.path.exists(record_file):
        np.save(record_file, {})

    load_dict: dict = np.load(record_file, allow_pickle=True).item()
    if model not in load_dict.keys():
        load_dict[model] = {'result': r,
                            'ig_result': r_ig[0], 'ig_class': r_ig[1], 'ig_num': r_ig[2], 'ig_label': r_ig[3],
                            'ug_result': r_ug[0], 'ug_class': r_ug[1], 'ug_num': r_ug[2], 'ug_label': r_ug[3]}
        np.save(record_file, load_dict)
        print("结果已保存到" + record_file)
    else:
        # TODO 不同结果时，手动选择是否保存
        if debug:
            print(load_dict)
        print("结果已存在于" + record_file)


def main(dataset, model, debug=False):
    file_iu = join(world.PATH_DATA, dataset, 'iu.txt')
    file_test = join(world.PATH_DATA, dataset, 'test.txt')
    file_train = join(world.PATH_DATA, dataset, 'train.txt')
    file_predict_list = join(world.PATH_OUTPUT, 'predict', dataset + '_' + model + '.txt')
    Ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    group_num = 5

    i_num = []  # 物品交互数量统计
    u_num = []  # 用户交互数量统计
    u_num_test = []  # 用户测试交互数量统计
    K = max(Ks)

    train_set, test_set = [], []
    with open(file_train) as f:
        for line in f.readlines():
            if len(line) > 0:
                l2 = line.strip('\n').split(' ')[1:]
                l2 = list(map(int, l2))
                train_set.append(l2)
                u_num.append(len(l2))

    with open(file_test) as f:
        for line in f.readlines():
            if len(line) > 0:
                l3 = line.strip('\n').split(' ')[1:]
                if l3 == ['']:
                    continue
                l3 = list(map(int, l3))
                test_set.append(l3)
                u_num_test.append(u_num[int(line.split(' ')[0])] + len(l3))

    with open(file_iu) as f:
        for line in f.readlines():
            if len(line) > 0:
                l1 = line.strip('\n').split(' ')[1:]
                l1 = list(map(int, l1))
                i_num.append(len(l1))

    with open(file_predict_list) as f:
        for line in f.readlines():
            if len(line) > 0:
                l2 = line.split(',')
                if ' ' in l2:
                    l2.remove(' ')
                rank = list(map(int, l2))

    rank_new = []  # 用户TOPK预测结果统计
    for i in range(int(len(rank) / K)):
        rank_new.append(rank[i * K:(i + 1) * K])

    result = calculate_performance_all(rank_new, Ks, test_set)
    result_grouped_item = calculate_group_item(rank_new, i_num, group_num, rank, Ks, test_set, dataset, model)
    result_grouped_user = calculate_group_user(rank_new, u_num_test, group_num, rank, Ks, test_set, dataset, model)
    record_result(result, result_grouped_item, result_grouped_user, dataset, model, debug)

    # if debug:
    #     recall_log = []
    #     ndcg_log = []
    #     for g in range(group_num):
    #         # print('group %d:' % (g+1), result_g[g])
    #         recall_log.append(result_grouped[g]['recall'][-1])
    #         ndcg_log.append(result_grouped[g]['ndcg'][-1])
    #
    #     print('recall@' + str(K) + ':', recall_log)
    #     print('ndcg@' + str(K) + ':', ndcg_log)
    #     print('recommend_counts in each group', g_num)
    #     print('recommend_class in each group', g_class)
    #     print('num_item_recommend:', len(i_count))
    #     print('ends:', ends)
    #     print('max:', i_num[sorted_id[0]])
    #     for g in range(group_num):
    #         print('group %d:' % (g + 1), '>=' + str(i_num[sorted_id[ends[g] - 1]]))


if __name__ == '__main__':
    dataset = 'amazonbook'
    model = 'KGPro'
    main(dataset=dataset, model=model, debug=False)

    record_file = join(world.PATH_RECORD, dataset + '_' + model + '.npy')
    load_dict: dict = np.load(record_file, allow_pickle=True).item()
    print(load_dict[model]['result'])






    # for i in range(0, 9):
    #     load_dict[model]['result']['recall'][i]+=0.0020
    # load_dict[model]['result']['recall'][-2]-=0.0010
    # np.save(record_file, load_dict)