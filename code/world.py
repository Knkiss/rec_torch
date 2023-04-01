# -*- coding: utf-8 -*-
"""
@Project ：rec_torch
@File    ：world.py
@Author  ：Knkiss
@Date    ：2023/2/14 9:59
"""
import argparse
import platform
from os.path import join

import torch

# region 模型参数设置
# region 推荐
seed = 2020
epoch = 0
TRAIN_epochs = 1000
embedding_dim = 64
topKs = [10, 20]
decay = 1e-4
root_model = False
# endregion

# region SGL
ssl_temp = 0.2  # 对比loss温度系数
ssl_reg = 0.1  # 对比loss比例
ssl_ratio = 0.5  # 图生成比例
SGL_RATIO = 0.5  # 图生成比例
# endregion

# region KGCL control
entity_num_per_item = 10  # 一个item取多少个entity
kg_p_drop = 0.5  # kg去边概率
ui_p_drop = 0.1  # ui去边概率
# endregion

# region SSM Loss
SSM_Loss_enable = False
SSM_Loss_cos = True     # True=cos False=内积
SSM_Loss_temp = 0.1     # 温度系数 越小对正负例区分越大
# endregion
# endregion

# region 命令行参数读取


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KGCL',
                        help="[MF, LightGCN, SGL, QKV, GraphCL, KGCL, KGCL_my]")
    parser.add_argument('--dataset', type=str, default='lastfm_big',
                        help="[MIND, amazonbook, movielens1m, yelp2018, citeulikea, lastfm]")
    parser.add_argument('--metrics', type=list, default=['Precision', 'NDCG', 'Recall'],
                        help="[Recall, Precision, NDCG]")
    parser.add_argument('--train_batch', type=int, default=2048)
    parser.add_argument('--test_batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nohup', type=bool, default=False)
    return parser.parse_args()


args = parse_args()
model = args.model
dataset = args.dataset
metrics = args.metrics
learning_rate = args.lr
train_batch_size = args.train_batch
test_u_batch_size = args.test_batch
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
# endregion

# region 功能设置
ROOT_PATH = "F:/Code/MINE/rec_torch"
if platform.system().lower() == 'linux':
    ROOT_PATH = "/home/byl/code/rec_torch/"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
OUTPUT_PATH = join(CODE_PATH, 'output')
PRETRAIN_PATH = join(OUTPUT_PATH, 'pretrain')
BOARD_PATH = join(OUTPUT_PATH, 'tensorboard_cache')

tensorboard_enable = False  # 使用tensorboard
tensorboard_instance = None

early_stop_enable = True  # 早停启用
early_stop_epoch_cnt = 15  # 早停计数器
early_stop_metric = metrics[-1]
test_start_epoch = 25  # 测试开始epoch
test_verbose_epoch = 1  # 测试间隔epoch

pretrain_input_enable = False  # 使用预训练Emb
pretrain_output_enable = False  # 保存当前模型Emb
pretrain_input = 'lightGCN'  # 预训练Emb文件名

mail_on_stop_enable = False  # 程序运行结束时发送邮件
mail_host = 'smtp.qq.com'
mail_user = '962443828'
mail_pass = 'jbmsrsjphuhgbfgd'
mail_sender = '962443828@qq.com'
mail_receivers = ['962443828@qq.com']
mail_comment = ''

linux_nohup = args.nohup
tqdm_enable = True
if linux_nohup:
    tqdm_enable = False
    mail_on_stop_enable = True
# endregion

# region 数据集设置
if dataset == 'MIND':
    pass
elif dataset == 'amazonbook':
    test_start_epoch = 5
    early_stop_epoch_cnt = 10
    # ui_p_drop = 0.05
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
elif dataset == 'lastfm_big':
    test_start_epoch = 5
    early_stop_epoch_cnt = 10
# endregion

# region 模型设置
if model == 'KGCL':
    pretrain_input_enable = False
    pretrain_input = 'KGCL'
elif model == 'QKV':
    pretrain_input_enable = True
    pretrain_input = 'lightGCN'
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


print('\n--------------------- Settings ---------------------')
i = 0
a = globals()
for i in a:
    if isinstance(a[i], (float, str, int, list, bool)):
        if a[i] == 'i' or i.__contains__('__'):
            continue
        print(i + ": " + str(a[i]))
print('----------------------------------------------------')
