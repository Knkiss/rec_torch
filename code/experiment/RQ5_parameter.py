# -*- coding: utf-8 -*-
import os

import world
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


marker_list = [m for m, func in Line2D.markers.items()
               if func != 'nothing' and m not in Line2D.filled_markers] + list(Line2D.filled_markers)
color_list = ['purple', 'y', 'coral', 'c', 'b', 'g', 'm', 'orange', 'r']


def show2dParameters(x, data, label, names, metric, dataset):
    colorA = '#EBC246'
    colorB = '#EB5446'
    colorC = '#4A53F7'

    plt.plot(x, data[0], marker='d', ms=10, lw=3, color=colorA, label="$\lambda_1 = 0.05$", ls='-.')
    plt.plot(x, data[1], marker='s', ms=10, lw=3, color=colorB, label="$\lambda_1 = 0.1$")
    plt.plot(x, data[2], marker='o', ms=10, lw=3, color=colorC, label="$\lambda_1 = 0.2$", ls='--')

    plt.legend(fontsize=14, loc='lower right')
    plt.grid(linestyle='--')
    plt.xticks(x, names, fontsize=16, weight='bold')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel(r'Number of item groups $n_{g}$', fontsize=18, weight='bold')  # X轴标签
    plt.ylabel(metric+"@20", fontsize=18, weight='bold')

    # # amazonbook
    if dataset == 'amazonbook':
        plt.yticks([0.13, 0.14, 0.15, 0.16, 0.17], fontsize=16, weight='bold')
        plt.text(2, np.max(data[1]) + 0.002, '0.1652', family='monospace', fontsize=16, color=colorB)
        plt.text(3, np.max(data[1]) + 0.001, '0.1645', family='monospace', fontsize=16, color=colorB)
        plt.title("AmazonBook", fontsize=18, weight='bold')  # 数据集标题
    elif dataset == 'movielens1m_kg':
        plt.yticks([0.31, 0.32, 0.33, 0.34, 0.35, 0.36], fontsize=16, weight='bold')
        plt.text(2, np.max(data[1]) + 0.002, '0.3502', family='monospace', fontsize=16, color=colorB)
        plt.text(1, np.max(data[2]) + 0.006, '0.3497', family='monospace', fontsize=16, color=colorB)
        plt.title("Movielens-1M", fontsize=18, weight='bold')  # 数据集标题
    elif dataset == 'lastfm_kg':
        plt.text(2, np.max(data[1]) + 0.002, '0.4000', family='monospace', fontsize=16, color=colorB)
        plt.text(1, np.max(data[2]) + 0.004, '0.3961', family='monospace', fontsize=16, color=colorB)
        plt.yticks([0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41], fontsize=16, weight='bold')
        plt.title("LastFM", fontsize=18, weight='bold')  # 数据集标题

    # plt.show()
    output_dir = os.path.join(world.PATH_PLOT, 'WORK2', 'RQ5')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, dataset + '_parameter.eps'), dpi=900, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, dataset + '_parameter.png'), dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    dataset = 'amazonbook'  # amazonbook、movielens1m_kg、lastfm_kg
    dim_1 = 'hyper_WORK2_cluster_num'  # X轴上坐标
    dim_2 = 'hyper_SSM_Regulation'  # 不同的图例即线
    metric = 'Recall'  # Precision、NDCG、Recall

    dim_1_select = ['0.05', '0.1', '0.2']
    dim_2_select = ['3', '5', '10', '20', '50']

    data = {}
    with open('RQ5/' + dataset + '.txt') as f:
        for i in f.readlines():
            result = eval(i.replace("array", "list"))

            if result[dim_2] not in data.keys():
                data[result[dim_2]] = {}
            data[result[dim_2]][result[dim_1]] = result[metric][-1]

    show_data = []
    for i in dim_1_select:
        line_data = []
        for j in dim_2_select:
            print(data)
            print(i, j)
            line_data.append(data[str(i)][float(j)])
        show_data.append(line_data)

    show2dParameters(x=range(len(dim_2_select)), data=show_data, label=dim_1_select, names=dim_2_select, metric=metric, dataset=dataset)
