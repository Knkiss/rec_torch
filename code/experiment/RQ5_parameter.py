# -*- coding: utf-8 -*-
import os

import world
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


marker_list = [m for m, func in Line2D.markers.items()
               if func != 'nothing' and m not in Line2D.filled_markers] + list(Line2D.filled_markers)
color_list = ['purple', 'y', 'coral', 'c', 'b', 'g', 'm', 'orange', 'r']


def show2dParameters(x, data, label, names, metric):
    colorA = '#EBC246'
    colorB = '#EB5446'
    colorC = '#4A53F7'

    plt.plot(x, data[0], marker='d', ms=10, lw=3, color=colorA, label="$\lambda_2 / \lambda_1 = 0.01$", ls='-.')
    plt.plot(x, data[1], marker='s', ms=10, lw=3, color=colorB, label="$\lambda_2 / \lambda_1 = 0.05$")
    plt.plot(x, data[2], marker='o', ms=10, lw=3, color=colorC, label="$\lambda_2 / \lambda_1 = 0.1$", ls='--')

    plt.legend(fontsize=14, loc='lower right')
    plt.grid(linestyle='--')
    plt.xticks(x, names, fontsize=16, weight='bold')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel(r'Temperature $\tau$', fontsize=18, weight='bold')  # X轴标签
    plt.ylabel(metric+"@20", fontsize=18, weight='bold')

    # # amazonbook
    # plt.yticks([0.12, 0.13, 0.14, 0.15, 0.16], fontsize=16, weight='bold')
    # plt.text(1, np.max(data[1]) + 0.002, '0.1519', family='monospace', fontsize=16, color=colorB)
    # plt.text(2, np.max(data[2]) + 0.002, '0.1517', family='monospace', fontsize=16, color=colorC)
    # plt.title("AmazonBook", fontsize=18, weight='bold')  # 数据集标题

    # # movielens
    # plt.yticks([0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35], fontsize=16, weight='bold')
    # plt.text(1, np.max(data[1]) + 0.002, '0.3351', family='monospace', fontsize=16, color=colorB)
    # plt.text(2, np.max(data[2]) + 0.002, '0.3384', family='monospace', fontsize=16, color=colorC)
    # plt.title("Movielens-1M", fontsize=18, weight='bold')  # 数据集标题

    # lastfm
    plt.text(1.2, np.max(data[1]) + 0.001, '0.3860', family='monospace', fontsize=16, color=colorB)
    plt.text(2, np.max(data[2]) + 0.002, '0.3870', family='monospace', fontsize=16, color=colorC)
    plt.yticks([0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40], fontsize=16, weight='bold')
    plt.title("LastFM", fontsize=18, weight='bold')  # 数据集标题

    # plt.show()
    output_dir = os.path.join(world.PATH_PLOT, 'KGAG', 'RQ5')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, dataset + '_parameter.eps'), dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    dataset = 'lastfm_kg'  # amazonbook、movielens1m_kg、lastfm_kg
    dim_1 = 'hyper_ssl_temp'  # X轴上坐标
    dim_2 = 'hyper_ssl_reg'  # 不同的图例即线
    metric = 'Recall'  # Precision、NDCG、Recall

    dim_1_select = ['0.01', '0.05', '0.1', '0.2', '0.5']
    dim_2_select = ['0.1', '0.15', '0.2', '0.25', '0.3']

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
            line_data.append(data[float(i)][float(j)])
        show_data.append(line_data)

    show2dParameters(x=range(len(dim_2_select)), data=show_data, label=dim_1_select, names=dim_2_select, metric=metric)
