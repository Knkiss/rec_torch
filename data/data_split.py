import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_dict_from_data(data):
    interaction_dict = {}
    for i in data:
        if i[2] == 1:
            if i[0] in interaction_dict.keys():
                interaction_dict[i[0]].append(i[1])
            else:
                interaction_dict[i[0]] = [i[1]]
    for i in interaction_dict.keys():
        interaction_dict[i].sort()

    return [(k, interaction_dict[k]) for k in sorted(interaction_dict.keys())]


def write_dict_to_file(interaction_dict, file):
    with open(file, mode='w') as f:
        for i in interaction_dict:
            f.write(str(i[0]) + " ")
            for j in i[1]:
                if j is i[1][-1]:
                    f.write(str(j))
                else:
                    f.write(str(j) + " ")
            f.write("\n")


def train_test_split(rating_file, test_ratio):
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)
    n_ratings = rating_np.shape[0]
    test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    left = set(range(n_ratings)) - set(test_indices)

    train_dict = get_dict_from_data(rating_np[list(left)])
    test_dict = get_dict_from_data(rating_np[test_indices])

    write_dict_to_file(train_dict, 'lastfm_kg/train.txt')
    write_dict_to_file(test_dict, 'lastfm_kg/test.txt')


def kg_split(kg_file):
    kg = np.loadtxt(kg_file, dtype=np.int64)
    with open('lastfm_kg/kg.txt', mode='w') as f:
        for i in kg:
            f.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")


def ui_to_iu(input_train, input_test, output_iu):
    all_iu = {}
    with open(input_train, mode='r') as f:
        for i in f.readlines():
            i = i.replace('\n', '').split(' ')
            uid = int(i[0])
            iid = i[1:]
            if iid == ['']:
                continue
            for j in iid:
                j = int(j)
                if j in all_iu.keys():
                    if uid not in all_iu[j]:
                        all_iu[j].append(uid)
                else:
                    all_iu[j] = [uid]
    with open(input_test, mode='r') as f:
        for i in f.readlines():
            i = i.replace('\n', '').split(' ')
            uid = int(i[0])
            iid = i[1:]
            if iid == ['']:
                continue
            for j in iid:
                j = int(j)
                if j in all_iu.keys():
                    if uid not in all_iu[j]:
                        all_iu[j].append(uid)
                else:
                    all_iu[j] = [uid]
    with open(output_iu, mode='w') as f:
        for i in sorted(all_iu.keys()):
            f.write(str(i) + ' ')
            for uid in all_iu[i]:
                if uid is all_iu[i][-1]:
                    f.write(str(uid))
                else:
                    f.write(str(uid) + ' ')
            f.write('\n')


def kg_resort(i_n, input_file, output_file):
    origin_kg_data = pd.read_csv(input_file, sep=' ', names=['h', 'r', 't'], engine='python')
    print(origin_kg_data.min(), origin_kg_data.max())

    h = origin_kg_data[origin_kg_data['h'] <= i_n]
    t = origin_kg_data[origin_kg_data['t'] <= i_n]
    h = h[h['t'] > i_n]
    t = t[t['h'] > i_n]
    t[['h', 't']] = t[['t', 'h']]
    process_kg_data = pd.concat([h, t])

    sub = pd.Series({'h': 0, 'r': 0, 't': i_n})
    process_kg_data = process_kg_data - sub
    process_kg_data = process_kg_data.sort_values(by=['h', 'r', 't'], ignore_index=True)

    # 对tail从0重新排序
    j = 0
    for i in tqdm(range(process_kg_data['t'].max() + 1)):
        if len(process_kg_data[process_kg_data['t'] == i]) > 0:
            process_kg_data.loc[process_kg_data['t'] == i, ['t']] = j
            j += 1

    # 对relation从0重新排序
    j = 0
    for i in tqdm(range(process_kg_data['r'].max() + 1)):
        if len(process_kg_data[process_kg_data['r'] == i]) > 0:
            process_kg_data.loc[process_kg_data['r'] == i, ['r']] = j
            j += 1

    process_kg_data.to_csv(output_file, sep=' ', header=False, index=False)


if __name__ == '__main__':
    dataset = 'movielens1m_kg'
    item_num = 2346

    file_train = dataset + '/train.txt'
    file_test = dataset + '/test.txt'
    file_iu = dataset + '/iu.txt'
    file_kg_graph = dataset + '/kg_graph.txt'
    file_kg = dataset + '/kg.txt'

    # train_test_split('ratings_final', test_ratio=0.2)
    # kg_split('kg_final.txt')
    # ui_to_iu(input_train=file_train, input_test=file_test, output_iu=file_iu)

    kg_resort(item_num, file_kg, file_kg)
