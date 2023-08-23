# -*- coding: utf-8 -*-
# 颜色：c 青绿，m 洋红，k 黑色
# 形状：o,v,^,<,>,s,p,1,2,3,4,h,H,d,D,+,x<*
import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from decimal import Decimal

import RQ0_calculate
import world
import warnings

warnings.filterwarnings('ignore')

marker_list = [m for m, func in Line2D.markers.items()
               if func != 'nothing' and m not in Line2D.filled_markers] + list(Line2D.filled_markers)
color_list = ['purple', 'y', 'coral', 'c', 'b', 'g', 'm', 'orange', 'r']
metric_list = ['recall', 'ndcg']
metric_list_name = ['Recall', 'NDCG']
dataset_name = {'amazonbook': 'AmazonBook', 'movielens1m_kg': 'Movielens-1M', 'lastfm_kg': 'LastFM'}


def RQ3_compare_longTail(datasets, models, form='recall', type='png', debug=False):
    for dataset in datasets:
        name_list = []
        result_metric = {}
        result_class = {}
        result_num = {}
        for model in models:
            record_file = os.path.join(world.RECORD_PATH, dataset + '_' + model + '.npy')
            load_dict: dict = np.load(record_file, allow_pickle=True).item()
            if not name_list:
                for i in load_dict[model]['ig_label']:
                    name_list.append('>=' + str(i))
            result_metric[model] = []
            for i in range(len(name_list)):
                result_metric[model].append(load_dict[model]['ig_result'][i]['recall'][-1])
            result_class[model] = load_dict[model]['ig_class']
            result_num[model] = load_dict[model]['ig_num']
            pass

        if form == 'recall':
            data = result_metric
            ytitle = 'Recall@20'
        elif form == 'class':
            data = result_class
            ytitle = 'Item Class'
        elif form == 'num':
            data = result_num
            ytitle = 'Item Numbers'
        else:
            raise NotImplementedError("RQ3: 不存在的type类型")

        data_all = np.array(list(data.values()))
        data_min, data_max = np.min(data_all), np.max(data_all)

        x = [0.1, 0.3, 0.5, 0.7, 0.9]
        width = 0.04
        width1 = 0.05
        ax = plt.subplot(111)
        plt.bar(x, data[models[-1]][::-1], width=width, label=models[-1], fc='#70c1b3', edgecolor='k')
        for i in range(len(x)):
            x[i] += width1
        plt.bar(x, data[models[-2]][::-1], width=width, label=models[-2], tick_label=name_list, fc='#2a9d8f', edgecolor='k')
        for i in range(len(x)):
            x[i] += width1
        plt.bar(x, data[models[0]][::-1], width=width, label=models[0], fc='#1a535c', edgecolor='k')
        plt.yticks(fontsize=14, weight='bold')
        plt.xticks(size=14, weight='bold')
        # 标题
        plt.ylabel(ytitle, fontsize=18, weight='bold')  # Y轴标签
        plt.xlabel("Item Group", fontsize=18, weight='bold')  # Y轴标签
        plt.legend(fontsize=14, loc='upper right', ncol=1)
        ax.grid(axis='y', linestyle='--')
        ax.set_axisbelow(True)

        if data_max/data_min > 10:
            t = math.log10(data_min)
            scale = math.floor(t)
            plt.ylim([pow(10, scale), data_max*2])
            plt.yscale('log')

        plt.title(dataset, fontsize=20, weight='bold')

        output_dir = os.path.join(world.PLOT_PATH, 'RQ3')
        if debug:
            plt.show()
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, dataset + '_itemGroup_' + form + '.' + type), dpi=900, bbox_inches='tight')
            plt.close()
    print("RQ3：所有数据集长尾物品分组比较绘制完成")


def RQ1_compare_all(datasets, models, x_ticks, type='png', fig_show=False, fig_save=True,
                    table_dataset_show=True, table_metrics_show=False, table_latex_show=False):
    performance_table = {}
    for dataset in datasets:
        performance_table[dataset] = {}
        for metric in metric_list:
            performance_table[dataset][metric] = {}
            data = {}
            y_max, y_min = 0, 1

            for model in models:
                record_file = os.path.join(world.PATH_RECORD, dataset + '_' + model + '.npy')
                load_dict: dict = np.load(record_file, allow_pickle=True).item()
                data[model] = load_dict[model]['result'][metric]
                performance_table[dataset][metric][model] = data[model][-1]
                y_max = max(y_max, max(data[model]))
                y_min = min(y_min, min(data[model]))

            y_max = math.ceil((y_max + y_max/100) * 1000) / 1000
            y_min = math.floor((y_min - y_min/100) * 1000) / 1000
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
            plt.title(dataset_name[dataset], fontsize=15, weight='bold')

            output_dir = os.path.join(world.PATH_PLOT, 'RQ1')

            if fig_show:
                plt.show()
            elif fig_save:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, dataset + '_' + metric + '.' + type), dpi=900, bbox_inches='tight')
                plt.close()
    print("RQ1：所有数据集独立Top-N 绘制完成")

    if table_dataset_show:
        for dataset in datasets:
            print(dataset.ljust(15, ' '), end='')
            # print(''.ljust(10, ' '), end='')
            for i in metric_list_name:
                print(i.rjust(15, ' '), end=' ')
            print()
            for i in models:
                if i is models[0]:
                    print("Ours".ljust(15, ' '), end='')
                else:
                    print(i.ljust(15, ' '), end='')
                for j in metric_list:
                    print(str(round(performance_table[dataset][j][i] * 10000) / 10000).rjust(15, ' '), end=' ')
                print()
            print()

    if table_metrics_show:
        for metric in metric_list:
            print(metric.ljust(15, ' '), end='')
            for i in datasets:
                print(i.rjust(15, ' '), end=' ')
            print()
            for i in models:
                if i is models[0]:
                    print("Ours".ljust(15, ' '), end='')
                else:
                    print(i.ljust(15, ' '), end='')
                for j in datasets:
                    print(str(round(performance_table[j][metric][i] * 10000) / 10000).rjust(15, ' '), end=' ')
                print()
            print()

    if table_latex_show:
        # 统计数据最大值与次大值，并计算提升值
        for dataset in datasets:
            for metric in metric_list:
                data = {}
                for model in models:
                    data[model] = performance_table[dataset][metric][model]
                data = sorted(data.items(),  key=lambda d: d[1], reverse=True)
                performance_table[dataset][metric]['max'] = data[0][0]
                performance_table[dataset][metric]['second'] = data[1][0]
                if data[0][0] == models[0]:
                    performance_table[dataset][metric]['improve'] = (data[0][1] - data[1][1]) / data[1][1]
                else:
                    performance_table[dataset][metric]['improve'] = -1

        # 打印数据及抬头
        for i in datasets:
            if i != datasets[-1]:
                fenge = 'c|'
            else:
                fenge = 'c'
            print(" & \\multicolumn{" + str(len(metric_list)) + '}{' + fenge + '}{' + i + '}', end='')
        print(' \\\\')

        # 打印指标抬头
        for i in range(len(datasets)):
            for j in range(len(metric_list)):
                print(' & ' + metric_list_name[j], end='')
        print(' \\\\ \\hline')

        # 打印模型性能
        for model in models[::-1]:
            print(model, end='')
            for dataset in datasets:
                for metric in metric_list:
                    res_origin = performance_table[dataset][metric][model]
                    res_process = Decimal(res_origin).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
                    if model == performance_table[dataset][metric]['max']:
                        print(' & \\textbf{' + str(res_process), end='}')
                    elif model == performance_table[dataset][metric]['second']:
                        print(' & \\underline{' + str(res_process), end='}')
                    else:
                        print(' & ' + str(res_process), end='')
            print(' \\\\', end='')
            if model in models[0:2]:
                print(' \\hline')
            else:
                print('')

        # 打印模型性能增长幅度
        print("Improve", end='')
        for dataset in datasets:
            for metric in metric_list:
                imp_origin = performance_table[dataset][metric]['improve']
                imp_process = float(Decimal(imp_origin).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")) * 100
                print(' & ' + str(imp_process) + "\\%", end='')
        print(' \\\\')


def RQ0_calculate_all(datasets, models, debug=False):
    finish = True
    for (dataset, model) in itertools.product(datasets, models):
        file = os.path.join(world.PATH_RECORD, dataset + '_' + model + '.npy')
        if not os.path.exists(file):
            try:
                if debug:
                    print(file, '未存在，计算结果')
                RQ0_calculate.main(dataset, model)
            except Exception as e:
                print(e)
                print(file, '计算失败')
                finish = False
                continue
        elif debug:
            print(file, '已存在')
    if finish:
        print("RQ0：所有数据处理完毕")
    else:
        raise FileNotFoundError("RQ0：数据处理异常，请检查失败原因后重新运行")


if __name__ == '__main__':
    dataset_list = ['amazonbook', 'movielens1m_kg', 'lastfm_kg']
    model_list = ['Ours', 'KGCL', 'SGL', 'LightGCN', 'KGIN', 'MCCLK', 'KGAT', 'MF', 'KGCN']
    save_fig_type = 'eps'  # png 或 eps

    world.PATH_PLOT = os.path.join(world.PATH_PLOT, model_list[0])
    debug = False

    RQ0_calculate_all(dataset_list, model_list)
    RQ1_compare_all(dataset_list, model_list, x_ticks=range(2, 21, 2), type=save_fig_type, fig_show=True,
                    fig_save=False, table_dataset_show=False, table_metrics_show=False, table_latex_show=False)
    # RQ1_compare_all(dataset_list, model_list, x_ticks=range(2, 21, 2), type=save_fig_type, debug=debug)
    # RQ3_compare_longTail(datasets=dataset_list, models=model_list, form='class', type=save_fig_type, debug=debug)
    # RQ3_compare_longTail(datasets=dataset_list, models=model_list, form='num', type=save_fig_type, debug=debug)
    # RQ3_compare_longTail(datasets=dataset_list, models=model_list, form='recall', type=save_fig_type, debug=debug)
