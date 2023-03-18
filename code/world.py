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

# region QKV model version @default=False
use_linear = False
# endregion

# region KGCL model version @default=False
user_item_preference = False    # 提升
item_entity_random_walk = False
remove_Trans = False            # 提升

entity_num_per_item = 10  # 一个item取多少个entity
kg_p_drop = 0.5  # kg去边概率
ui_p_drop = 0.1  # ui去边概率
# endregion

# region SGL model parameter
ssl_temp = 0.2   # 对比loss温度系数
ssl_reg = 0.1    # 对比loss比例
ssl_ratio = 0.5  # 图生成比例
# endregion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    # read
    parser.add_argument('--model', type=str, default='KGCL',
                        help="available datasets: [KGCL, MF, lightGCN, SGL, QKV, GraphCL]")
    parser.add_argument('--dataset', type=str, default='amazonbook',
                        help="available datasets: [amazonbook, movielens1m, yelp2018, citeulikea, lastfm]")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")

    # train
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)

    # parameters
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--recDim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--keepProb', type=float, default=0.8,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--dropout', type=int, default=1,
                        help="using the dropout or not")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalization")

    # test
    parser.add_argument('--topKs', nargs='?', default="[10, 20]",
                        help="@k test list")
    parser.add_argument('--testBatch', type=int, default=4096,
                        help="the batch size of users for testing")

    # tensorboard
    parser.add_argument('--comment', type=str, default="")
    return parser.parse_args()


# region 命令行args读取
epoch = 0
args = parse_args()
model = args.model
dataset = args.dataset
TRAIN_epochs = args.epochs
comment = args.comment
seed = args.seed
decay = args.decay
topKs = eval(args.topKs)
config = {}
config['A_split'] = False
config['A_n_fold'] = args.a_fold
config['latent_dim_rec'] = args.recDim
config['lightGCN_n_layers'] = args.layer
config['keep_prob'] = args.keepProb
config['dropout'] = args.dropout
config['lr'] = args.lr
config['decay'] = args.decay
config['train_batch_size'] = args.bpr_batch
config['test_u_batch_size'] = args.testBatch
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
# endregion

# region 文件夹路径索引
ROOT_PATH = "D://byl//rec_torch"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'tensorboard_cache')
# endregion

# region tensorboard 结果可视化
tensorboard_enable = False             # 使用tensorboard
tensorboard_instance = None
# endregion

# region 测试与早停设置
early_stop_enable = True  # 早停启用
early_stop_epoch_cnt = 10  # 早停计数器
test_start_epoch = 25  # 测试开始epoch
test_verbose_epoch = 1  # 测试间隔epoch
# endregion

# region 预训练模型Emb加载和保存
pretrain_input_enable = False  # 使用预训练Emb
pretrain_output_enable = True  # 保存当前模型Emb
pretrain_input = 'lightGCN'  # 预训练Emb文件名
pretrain_folder = 'pretrain/'  # 预训练Emb文件夹名
# endregion

# region 邮件提醒相关设置
mail_on_stop_enable = False  # 程序运行结束时发送邮件
mail_host = 'smtp.qq.com'
mail_user = '962443828'
mail_pass = 'jbmsrsjphuhgbfgd'
mail_sender = '962443828@qq.com'
mail_receivers = ['962443828@qq.com']
# endregion

# region 数据集设置
if dataset == 'MIND':
    pass
elif dataset == 'amazonbook':
    test_start_epoch = 5
    early_stop_epoch_cnt = 10
elif dataset == 'yelp2018':
    test_start_epoch = 5
    early_stop_epoch_cnt = 10
elif dataset == 'movielens1m':
    test_start_epoch = 5
    early_stop_epoch_cnt = 10
elif dataset == 'citeulikea':
    test_start_epoch = 5
    early_stop_epoch_cnt = 30
elif dataset == 'lastfm':
    test_start_epoch = 5
    early_stop_epoch_cnt = 30
# endregion

# region 需要使用预训练模型设置
if model == 'QKV':
    pretrain_input_enable = True
    pretrain_input = 'SGL'
    test_start_epoch = 0
    early_stop_enable = True
    early_stop_epoch_cnt = 20
elif model == 'GraphCL':
    pretrain_input_enable = True
    pretrain_input = 'SGL'
    test_start_epoch = 0
    early_stop_enable = True
    early_stop_epoch_cnt = 20
# endregion
