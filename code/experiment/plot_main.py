# -*- coding: utf-8 -*-
# 颜色：c 青绿，m 洋红，k 黑色
# 形状：o,v,^,<,>,s,p,1,2,3,4,h,H,d,D,+,x<*
import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np

import RQ0_calculate
import world
import warnings

warnings.filterwarnings('ignore')

marker_list = ['d', '>', 'o', '*', '^', 'v', '+', 's']
color_list = ['purple', 'y', 'coral', 'c', 'b', 'g', 'm', 'r']
metric_list = ['recall', 'precision', 'ndcg', 'map', 'hit_ratio']
metric_list_name = ['Recall', 'Precision', 'NDCG', 'MAP', 'Hit Ratio']


def RQ1_compare_all(datasets, models, x_ticks, type='png', debug=False):
    for dataset in datasets:
        for metric in metric_list:
            data = {}
            y_max, y_min = 0, 1

            for model in models:
                record_file = os.path.join(world.RECORD_PATH, dataset + '_' + model + '.npy')
                load_dict: dict = np.load(record_file, allow_pickle=True).item()
                data[model] = load_dict[model]['result'][metric]
                y_max = max(y_max, max(data[model]))
                y_min = min(y_min, min(data[model]))

            y_max = math.ceil(y_max * 1000) / 1000
            y_min = math.floor(y_min * 1000) / 1000
            y_step = math.ceil((y_max - y_min) / 0.006) / 1000
            y_ticks = [math.ceil((y * y_step + y_min) * 1000) / 1000 for y in range(7)]

            x_ticks = [str(x) for x in list(x_ticks)]
            x = range(len(x_ticks))
            for model in range(len(models)):
                key = models[model]
                if model == 0:
                    plt.plot(x, data[key], marker=marker_list[-1], ms=5, color=color_list[-1], label=key)
                else:
                    plt.plot(x, data[key], marker=marker_list[model], ms=5, color=color_list[model], label=key)

            plt.legend(fontsize=12, loc="upper left")
            plt.grid(linestyle='--')
            plt.xticks(x, x_ticks, fontsize=12, weight='bold')
            plt.margins(0)
            plt.subplots_adjust(bottom=0.10)
            plt.xlabel('Top-N', fontsize=13, weight='bold')
            plt.ylabel(metric_list_name[metric_list.index(metric)] + '@N', fontsize=13, weight='bold')
            plt.yticks(y_ticks, fontsize=12, weight='bold')
            plt.title(dataset, fontsize=15, weight='bold')

            output_dir = os.path.join(world.PLOT_PATH, 'RQ1')

            if debug:
                plt.show()
            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, dataset + '_' + metric + '.' + type), dpi=900, bbox_inches='tight')
                plt.close()
    print("RQ1：所有数据集独立Top-N 绘制完成")


def RQ0_calculate_all(datasets, models, debug=False):
    finish = True
    for (dataset, model) in itertools.product(datasets, models):
        file = os.path.join(world.RECORD_PATH, dataset + '_' + model + '.npy')
        if not os.path.exists(file):
            try:
                if debug:
                    print(file, '未存在，计算结果')
                RQ0_calculate.main(dataset, model)
            except Exception:
                print(file, '计算失败')
                finish = False
                continue
        elif debug:
            print(file, '已存在')
    if finish:
        print("RQ0：所有数据处理完毕，开始画图")
    else:
        raise FileNotFoundError("数据处理异常，请检查")


if __name__ == '__main__':
    dataset_list = ['amazonbook', 'bookcrossing', 'lastfm_kg', 'movielens1m_kg', 'yelp2018_kg']
    model_list = ['KGCL_my', 'KGCL', 'SGL', 'LightGCN', 'MF']
    debug = False
    save_fig_type = 'png'

    world.PLOT_PATH = os.path.join(world.PLOT_PATH, model_list[0])
    RQ0_calculate_all(dataset_list, model_list)
    RQ1_compare_all(dataset_list, model_list, x_ticks=range(2, 21, 2), type=save_fig_type, debug=debug)
