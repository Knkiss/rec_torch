# -*- coding: utf-8 -*-
"""
@Project ：rec_torch
@File    ：world.py
@Author  ：Knkiss
@Date    ：2023/2/14 9:59
"""
import argparse
from os.path import join
import torch


# KGCL model version
user_item_preference = False
item_entity_random_walk = False
use_Trans = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    # read
    parser.add_argument('--model', type=str, default='KGCL',
                        help="available datasets: [KGCL, MF, lightGCN, GraphCL]")
    parser.add_argument('--dataset', type=str, default='yelp2018',
                        help="available datasets: [yelp2018, amazon-book, MIND]")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")

    # train
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)

    # parameters
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--keepprob', type=float, default=0.8,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--dropout', type=int, default=1,
                        help="using the dropout or not")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")

    # test
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--multicore', type=int, default=0,
                        help='whether we use multiprocessing or not in test')
    parser.add_argument('--testbatch', type=int, default=4096,
                        help="the batch size of users for testing")

    # tensorboard
    parser.add_argument('--comment', type=str, default="")
    return parser.parse_args()


ROOT_PATH = "D://byl//rec_torch"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

tensorboard = True

entity_num_per_item = 10  # 一个item取多少个entity
kgc_temp = 0.2
kg_p_drop = 0.5  # kg去边概率
ui_p_drop = 0.1  # ui去边概率
test_start_epoch = 25  # 测试开始epoch
test_verbose = 1
early_stop_cnt = 5  # 多少次性能无提升截止
ssl_reg = 0.1

args = parse_args()
model = args.model
dataset = args.dataset
TRAIN_epochs = args.epochs
comment = args.comment
seed = args.seed

config = {}
config['A_split'] = False
config['A_n_fold'] = args.a_fold
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['keep_prob'] = args.keepprob
config['dropout'] = args.dropout
config['lr'] = args.lr
config['decay'] = args.decay
config['bpr_batch_size'] = args.bpr_batch
config['multicore'] = args.multicore
config['test_u_batch_size'] = args.testbatch

uicontrast = "WEIGHTED"

topks = eval(args.topks)

use_ckg = False

# mail
mail_host = 'smtp.qq.com'
mail_user = '962443828'
mail_pass = 'jbmsrsjphuhgbfgd'
sender = '962443828@qq.com'
receivers = ['962443828@qq.com']

if dataset == 'MIND':
    pass

elif dataset == 'amazon-book':
    pass

elif dataset == 'yelp2018':
    config['dropout'] = 1
    config['keep_prob'] = 0.8
    uicontrast = "WEIGHTED"
    ui_p_drop = 0.1
    test_start_epoch = 25
    early_stop_cnt = 10

elif dataset == 'citeulike-a':
    config['dropout'] = 1
    test_start_epoch = 25
    early_stop_cnt = 10

elif dataset == 'lastfm':
    config['dropout'] = 1
    test_start_epoch = 25
    early_stop_cnt = 10
