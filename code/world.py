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
para_1 = 1
para_2 = 0.1
para_3 = 1

hyper_test_ratio = 1
hyper_test_ratio_2 = 1
hyper_KGIN_n = 4  # [2, 3, 4, 5]
hyper_KGAT_layers = [64]  # [64], [64, 64], [64, 64, 64], [64, 32], [64, 32, 16]
hyper_ssl_temp = 0.2  # 对比loss温度系数
hyper_ssl_reg = 0.1  # 对比loss比例
hyper_SGL_RATIO = 0.5  # 图生成比例
hyper_KGDataset_entity_num_per_item = 10  # 一个item取多少个entity
hyper_KGCL_kg_p_drop = 0.5  # kg去边概率
hyper_KGCL_ui_p_drop = 0.1  # ui去边概率
hyper_KGAG_ablated_model = 0  # optional=[0,2,3,4] 0=vanilla 2=Qu 3=Qi 4=concatQui
hyper_SSM_Loss_temp = 0.2  # 温度系数 越小对正负例区分越大
hyper_SSM_Regulation = 0.1  # BPR 和 SSM的比例系数，加在SSM前
hyper_SSM_Margin = 1
hyper_decay = 1e-4
hyper_embedding_dim = 64
hyper_KGRec_best_hyper_group = 1  # option=[1,2,3]

hyper_WORK2_ckg_layers = 3
hyper_WORK2_ui_layers = 3

hyper_WORK2_BPR_mode = 1  # 使用哪个图的结果作BPR LOSS和推荐，1=ui，2=ckg，3=sum
hyper_WORK2_SSM_mode = 2  # enable>0 使用哪个图的结果作SSM LOSS，1=ui，2=ckg，3=sum
hyper_WORK2_KD_mode = 1  # enable>0 使用哪种蒸馏方式

sys_seed = 2020
sys_epoch = 0
sys_max_epochs = 1000
sys_topKs = [10, 20]  # [2,4,6,8,10,12,14,16,18,20]
sys_root_model = False
sys_ablation_name = ''


# region 命令行参数读取
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='WORK2')
    # classic: LightGCN、MF、GraphDA
    # contrastive: SSM、SimGCL、SGL
    # KG-based: KGRec、KGCL、MCCLK、KGIN、KGAT、KGCN
    # mine: PCL、KGIC、WORK2
    # unUse: QKV、GraphCL、EmbeddingBox
    parser.add_argument('--dataset', type=str, default='lastfm_kg')
    # UI数据集: 'citeulikea', 'lastfm', 'movielens1m', 'yelp2018'
    # KG数据集: 'amazonbook', 'yelp2018_kg', 'bookcrossing', 'movielens1m_kg', 'lastfm_kg', 'lastfm_wxkg'
    # GJJ的数据集: 'citeulikea_GJJ', 'lastfm_GJJ', 'movielens1m_GJJ'

    # PCL文章使用：'amazonbook', 'lastfm'
    # KGAG文章使用：'amazonbook', 'movielens1m_kg', 'lastfm_kg'
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
    parser.add_argument('--predict_list', type=bool, default=False)  # 是否保存推荐列表
    parser.add_argument('--time_calculate', type=bool, default=False)  # 是否开启时间统计

    parser.add_argument('--uuK', type=int, default=0)
    parser.add_argument('--iiK', type=int, default=0)
    parser.add_argument('--uiK', type=int, default=0)
    parser.add_argument('--iuK', type=int, default=0)

    parser.add_argument('--hyper1', type=int, default=10)
    parser.add_argument('--hyper2', type=float, default=0.001)

    return parser.parse_args()


args = parse_args()
WORK2_sample_uuK = args.uuK
WORK2_sample_iiK = args.iiK
WORK2_sample_uiK = args.uiK
WORK2_sample_iuK = args.iuK
hyper_WORK2_cluster_num = args.hyper1
hyper_KD_regulation = args.hyper2
hyper_WORK2_reset_ui_graph = (WORK2_sample_uuK + WORK2_sample_iiK + WORK2_sample_uiK + WORK2_sample_iuK) > 0
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
PATH_ROOT = "D:/Codes/rec_torch_main"
if platform.system().lower() == 'linux':
    PATH_ROOT = "/home/byl/code/rec_torch_main"
PATH_CODE = join(PATH_ROOT, 'code')
PATH_DATA = join(PATH_ROOT, 'data')
PATH_OUTPUT = join(PATH_ROOT, 'output')
PATH_PRETRAIN = join(PATH_OUTPUT, 'pretrain')
PATH_BOARD = join(PATH_OUTPUT, 'tensorboard_cache')
PATH_PREDICT = join(PATH_OUTPUT, 'predict')
PATH_RECORD = join(PATH_OUTPUT, 'record')
PATH_PLOT = join(PATH_OUTPUT, 'plot')

tensorboard_instance = None

early_stop_epoch_cnt = 25  # 早停计数器
early_stop_metric = metrics[-1]
test_start_epoch = 1  # 测试开始epoch
test_verbose_epoch = 1  # 测试间隔epoch

pretrain_input_enable = False  # 使用预训练Emb
pretrain_output_enable = False  # 保存当前模型Emb
pretrain_input = 'LightGCN'  # 预训练Emb文件名

mail_host = 'smtp.qq.com'
mail_user = '962443828'
mail_pass = 'jbmsrsjphuhgbfgd'
mail_sender = '962443828@qq.com'
mail_receivers = ['962443828@qq.com']
mail_comment = ''

linux_nohup = args.nohup
epoch_result_show_enable = True
model_info_show_enable = True
dataset_info_show_enable = True
tqdm_enable = True
if linux_nohup:
    tqdm_enable = False
    mail_on_stop_enable = True
# endregion

if model == 'MF':
    early_stop_epoch_cnt = 30

if model == 'KGIN':
    if dataset in ['yelp2018kg']:
        hyper_KGIN_n = 2
    elif dataset in ['amazonbook', 'bookcrossing', 'movielens1m_kg']:
        hyper_KGIN_n = 3
    elif dataset in ['lastfm_wxkg', 'lastfm_kg']:
        hyper_KGIN_n = 4


def print_arguments():
    print('\n----------------------------------- Settings -----------------------------------')
    a = globals().copy()
    a = sorted(a.items(), key=lambda d: d[0])
    for i in a:
        if isinstance(i[1], (float, str, int, list, bool)):
            if i[0] == 'i' or '__' in i[0]:
                continue
            print(i[0].ljust(30) + ": " + str(i[1]).ljust(25))
    print('--------------------------------------------------------------------------------')
