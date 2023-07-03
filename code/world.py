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

test_ratio = 1
test_ratio_2 = 1

KGIN_n = 4  # [2, 3, 4, 5]
KGAT_layers = [64]  # [64], [64, 64], [64, 64, 64], [64, 32], [64, 32, 16]

# region 模型参数设置
# region 推荐
seed = 2020
epoch = 0
TRAIN_epochs = 1000
embedding_dim = 64
# topKs = [2,4,6,8,10,12,14,16,18,20]
topKs = [10, 20]
decay = 1e-4
root_model = False
# endregion

ssl_temp = 0.2  # 对比loss温度系数
ssl_reg = 0.1  # 对比loss比例
SGL_RATIO = 0.5  # 图生成比例

KGDataset_entity_num_per_item = 10  # 一个item取多少个entity
KGCL_kg_p_drop = 0.5  # kg去边概率
KGCL_ui_p_drop = 0.1  # ui去边概率
KGCL_my_ablated_model = 0  # optional=[0,1]  1=双KG不起作用，得到的Cui均为1

SSM_Loss_temp = 0.2  # 温度系数 越小对正负例区分越大
SSM_Regulation = 0.1  # BPR 和 SSM的比例系数，加在SSM前
SSM_Margin = 1


# endregion

# region 命令行参数读取
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KGIN')
    # classic: MF、LightGCN
    # contrastive: SGL、SSM、SimGCL
    # KG-based: KGCL、KGIN、KGAT
    # mine: PCL、KGCL_my
    # unUse: QKV、GraphCL
    parser.add_argument('--dataset', type=str, default='amazonbook')
    # UI数据集: 'citeulikea', 'lastfm', 'movielens1m', 'yelp2018'
    # KG数据集: 'amazonbook', 'yelp2018_kg', 'bookcrossing', 'movielens1m_kg', 'lastfm_kg', 'lastfm_wxkg'

    # PCL文章使用：'amazonbook', 'lastfm'
    parser.add_argument('--metrics', type=list, default=['Precision', 'NDCG', 'Recall'],
                        help="[Recall, Precision, NDCG]")
    parser.add_argument('--train_batch', type=int, default=2048)
    parser.add_argument('--test_batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nohup', type=bool, default=False)
    parser.add_argument('--tensorboard', type=bool, default=False)  # 是否记录为可视化
    parser.add_argument('--searcher', type=bool, default=False)  # 是否使用参数搜索
    parser.add_argument('--early_stop', type=bool, default=True)  # 早停是否开启
    parser.add_argument('--mail_on_stop', type=bool, default=False)  # 程序运行结束时是否发送邮件
    parser.add_argument('--predict_list', type=bool, default=True)  # 是否保存推荐列表
    parser.add_argument('--time_calculate', type=bool, default=False)  # 是否开启时间统计
    return parser.parse_args()


args = parse_args()
model = args.model
dataset = args.dataset
metrics = args.metrics
learning_rate = args.lr
train_batch_size = args.train_batch
test_u_batch_size = args.test_batch
tensorboard_enable = args.tensorboard
searcher = args.searcher
early_stop_enable = args.early_stop
mail_on_stop_enable = args.mail_on_stop
predict_list_enable = args.predict_list
time_calculate_enable = args.time_calculate
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
# endregion

# region 功能设置
ROOT_PATH = "F:/Code/MINE/rec_torch"
if platform.system().lower() == 'linux':
    ROOT_PATH = "/home/byl/code/rec_torch/"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
OUTPUT_PATH = join(ROOT_PATH, 'output')
PRETRAIN_PATH = join(OUTPUT_PATH, 'pretrain')
BOARD_PATH = join(OUTPUT_PATH, 'tensorboard_cache')
PREDICT_PATH = join(OUTPUT_PATH, 'predict')
RECORD_PATH = join(OUTPUT_PATH, 'record')
PLOT_PATH = join(OUTPUT_PATH, 'plot')

tensorboard_instance = None

early_stop_epoch_cnt = 15  # 早停计数器
early_stop_metric = metrics[-1]
test_start_epoch = 1  # 测试开始epoch
test_verbose_epoch = 1  # 测试间隔epoch

pretrain_input_enable = False  # 使用预训练Emb
pretrain_output_enable = False  # 保存当前模型Emb
pretrain_input = 'lightGCN'  # 预训练Emb文件名

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

if model == 'MF':
    early_stop_epoch_cnt = 30

if model == 'KGIN':
    if dataset in ['yelp2018kg']:
        KGIN_n = 2
    elif dataset in ['amazonbook', 'bookcrossing', 'movielens1m_kg']:
        KGIN_n = 3
    elif dataset in ['lastfm_wxkg', 'lastfm_kg']:
        KGIN_n = 4


def print_arguments():
    print('\n--------------------- Settings ---------------------')
    a = globals()
    for i in a:
        if isinstance(a[i], (float, str, int, list, bool)):
            if a[i] == 'i' or i.__contains__('__'):
                continue
            print(i + ": " + str(a[i]))
    print('----------------------------------------------------')
