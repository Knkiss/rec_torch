import os

import numpy as np


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


if __name__ == '__main__':
    train_test_split('ratings_final', test_ratio=0.2)
    kg_split('kg_final.txt')