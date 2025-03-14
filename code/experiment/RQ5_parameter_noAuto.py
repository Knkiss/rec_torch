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

    # 单独划一条线
    plt.plot(x, data, marker='s', ms=10, lw=3, color=colorB, label="$\lambda_1 = 0.1$")

    # 多条线
    # plt.plot(x, data[0], marker='d', ms=10, lw=3, color=colorA, label="$\lambda_1 = 0.05$", ls='-.')
    # plt.plot(x, data[1], marker='s', ms=10, lw=3, color=colorB, label="$\lambda_1 = 0.1$")
    # plt.plot(x, data[2], marker='o', ms=10, lw=3, color=colorC, label="$\lambda_1 = 0.2$", ls='--')
    # plt.legend(fontsize=14, loc='lower right')

    plt.grid(linestyle='--')
    plt.xticks(x, names, fontsize=16, weight='bold')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    # plt.xlabel(r'Number of item groups $n_{g}$', fontsize=18, weight='bold')  # X轴标签
    # plt.xlabel(r'Temperature ${\tau}_{pcl}$', fontsize=18, weight='bold')  # X轴标签
    # plt.xlabel(r'${K}$', fontsize=18, weight='bold')  # X轴标签  WORK3 K
    plt.xlabel(r'${\phi}$', fontsize=18, weight='bold')  # X轴标签  WORK3 K
    plt.ylabel(metric+"@20", fontsize=18, weight='bold')

    # # amazonbook
    if dataset == 'amazonbook':
        # plt.yticks([0.15, 0.155, 0.16, 0.165, 0.17], fontsize=16, weight='bold')
        # plt.yticks([0.162, 0.163, 0.164, 0.165, 0.166], fontsize=16, weight='bold')
        # plt.yticks([0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17], fontsize=16, weight='bold')  # WORK3 K
        # plt.yticks([0.15, 0.155, 0.16, 0.165, 0.17], fontsize=16, weight='bold')  # WORK3 tau
        plt.yticks([0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17], fontsize=16, weight='bold')  # WORK3 phi
        # plt.text(2, np.max(data[1]) + 0.002, '0.1652', family='monospace', fontsize=16, color=colorB)
        # plt.text(3, np.max(data[1]) + 0.001, '0.1645', family='monospace', fontsize=16, color=colorB)
        plt.title("AmazonBook", fontsize=18, weight='bold')  # 数据集标题
    elif dataset == 'movielens1m_kg':
        # plt.yticks([0.346, 0.347, 0.348, 0.349, 0.35, 0.351, 0.352], fontsize=16, weight='bold')
        # plt.yticks([0.344, 0.346, 0.348, 0.35, 0.352], fontsize=16, weight='bold')
        # plt.yticks([0.33, 0.335, 0.34, 0.345, 0.35, 0.355], fontsize=16, weight='bold')  # WORK3 K
        plt.yticks([0.285, 0.295,0.305,0.315,0.325,0.335,0.345,0.355], fontsize=16, weight='bold')  # WORK3 phi
        # plt.text(2, np.max(data[1]) + 0.002, '0.3502', family='monospace', fontsize=16, color=colorB)
        # plt.text(1, np.max(data[2]) + 0.006, '0.3497', family='monospace', fontsize=16, color=colorB)
        plt.title("Movielens-1M", fontsize=18, weight='bold')  # 数据集标题
    elif dataset == 'lastfm_kg':
        # plt.yticks([0.37, 0.38, 0.39, 0.40, 0.41], fontsize=16, weight='bold')
        plt.yticks([0.2, 0.25, 0.30, 0.35, 0.40, 0.45], fontsize=16, weight='bold')  # WORK3 K
        # plt.yticks([0.39,0.392,0.394,0.396,0.398,0.40], fontsize=16, weight='bold')  # WORK3 tau
        # plt.text(3, np.max(data[1]) + 0.003, '0.4000', family='monospace', fontsize=16, color=colorB)
        # plt.text(2, np.max(data[1]) + 0.002, '0.3978', family='monospace', fontsize=16, color=colorB)
        plt.title("LastFM", fontsize=18, weight='bold')  # 数据集标题

    # plt.show()
    output_dir = os.path.join(world.PATH_PLOT, 'KGPro', 'RQ5')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # plt.show()
    plt.savefig(os.path.join(output_dir, dataset + '_parameter_phi.eps'), dpi=900, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, dataset + '_parameter_phi.png'), dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    dataset = 'lastfm_kg'  # amazonbook、movielens1m_kg、lastfm_kg
    metric = 'Recall'  # Precision、NDCG、Recall

    # # 不同fai时的结果
    dim_2_select = ['0.01', '0.05', '0.1', '0.5', '1']  # 横轴 ng
    if dataset == 'amazonbook':
        data = [0.1609, 0.1660, 0.1647, 0.1549, 0.1493]
    elif dataset == "movielens1m_kg":
        data = [0.3443, 0.3478, 0.3450, 0.3238, 0.2946]
    else:
        data = [0.3947, 0.3974, 0.3962, 0.2909, 0.2749]

    # # 不同tau时的结果
    # dim_2_select = ['0.5', '0.75', '1', '1.5', '5']  # 横轴 ng
    # if dataset == 'amazonbook':
    #     data = [0.1558, 0.1623, 0.1647, 0.1660, 0.1588]
    # elif dataset == "movielens1m_kg":
    #     data = [0.3398, 0.3437, 0.3452, 0.3478, 0.3446]
    # else:
    #     data = [0.3925, 0.3941, 0.3946, 0.3974, 0.3939]

    # 不同K时候的结果
    # dim_2_select = ['2', '5', '10', '20', '50']  # 横轴 temperature
    # if dataset == 'amazonbook':
    #     data = [0.1509, 0.1610, 0.1647, 0.1660, 0.1478]
    # elif dataset == "movielens1m_kg":
    #     data = [0.3441, 0.3464, 0.3475, 0.3478, 0.3380]
    # else:
    #     data = [0.3168, 0.3908, 0.3947, 0.3973, 0.2912]

    # 不同lambda1和lambda2时候的结果
    dim_1_select = ['0.05', '0.1', '0.15']  # 图例
    # dim_2_select = ['0.01', '0.1', '0.5', '1', '10']  # 横轴 temperature
    # if dataset == 'amazonbook':
    #     data = [
    #         [0.1589, 0.1594, 0.1620, 0.1638, 0.1619],
    #         [0.1618, 0.1634, 0.1648, 0.1652, 0.1614],
    #         [0.1565, 0.1568, 0.1596, 0.1564, 0.1594]
    #     ]
    # elif dataset == "movielens1m_kg":
    #     data = [
    #         [0.3420, 0.3433, 0.3483, 0.3486, 0.3413],
    #         [0.3446, 0.3488, 0.3502, 0.3502, 0.3480],
    #         [0.3346, 0.3398, 0.3435, 0.3458, 0.3396]
    #     ]
    # else:
    #     data = [
    #         [0.3668, 0.3768, 0.3957, 0.3838, 0.3821],
    #         [0.3784, 0.3908, 0.4000, 0.3978, 0.3854],
    #         [0.3594, 0.3629, 0.3772, 0.3746, 0.3749]
    #     ]

    show2dParameters(x=range(len(dim_2_select)), data=data, label=dim_1_select, names=dim_2_select, metric=metric, dataset=dataset)
