import os

import matplotlib.pyplot as plt

import world

dataset = 'lastfm_kg'  # amazonbook、movielens1m_kg、lastfm_kg
name_list = ['Recall', 'NDCG']
color_list = [
    "#9DB4D3",  # 淡蓝色，相对明亮
    "#7A93C1",  # 中等深度的蓝色
    "#5A73AF",  # 较深的蓝色
    "#3B548D",  # 深蓝色，接近黄昏天空的颜色
    "#1D365B",   # 非常深的蓝色，接近深夜天空的颜色
    "#001A40",
    "#bfdaff"
]


if dataset == 'amazonbook':
    KGAG = [0.1654, 0.1266]
    ablation1 = [0.1108, 0.0901]
    ablation2 = [0.1388, 0.1026]
    ablation3 = [0.1447, 0.1093]
    ablation4 = [0.1527, 0.1104]
    ablation5 = [0.1622, 0.1235]
elif dataset == 'movielens1m_kg':
    KGAG = [0.3502, 0.5707]
    ablation1 = [0.2204, 0.448]
    ablation5 = [0.3458, 0.5645]
    ablation2 = [0.3226, 0.5457]
    ablation3 = [0.3377, 0.5534]
    ablation4 = [0.3454, 0.564]
elif dataset == 'lastfm_kg':
    KGAG = [0.4, 0.3475]
    ablation1 = [0.3354, 0.3045]
    ablation2 = [0.369, 0.2964]
    ablation3 = [0.3802, 0.3306]
    ablation4 = [0.3944, 0.3403]
    ablation5 = [0.3961, 0.3423]



bar_width = 0.045
bar_and_space_width = 0.05
ax1_x = [0.175]
ax2_x = [0.575]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlim([0.1, 0.9])

if dataset == 'amazonbook':
    ax1.set_ylim([0.08, 0.17])
    # ax1.set_yticks([0.150, 0.1505, 0.151, 0.1515, 0.152])
    ax2.set_ylim([0.08, 0.17])
    # ax2.set_yticks([0.112, 0.115, 0.118, 0.121, 0.124, 0.127])
    pass
elif dataset == 'movielens1m_kg':
    ax1.set_ylim([0.2, 0.36])
    # ax1.set_yticks([0.310, 0.315, 0.32, 0.325, 0.33, 0.335, 0.340])
    ax2.set_ylim([0.3, 0.9])
    # ax2.set_yticks([0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
    pass
elif dataset == 'lastfm_kg':
    ax1.set_ylim([0.28, 0.42])
    # ax1.set_yticks([0.36, 0.37, 0.38, 0.39, 0.40, 0.405])
    ax2.set_ylim([0.28, 0.42])
    # ax2.set_yticks([0.29, 0.35])
    pass

ax1.tick_params(labelsize=12)
ax1.bar(ax1_x, ablation1[0], width=bar_width, fc=color_list[0])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation2[0], width=bar_width, fc=color_list[1])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation3[0], width=bar_width, fc=color_list[2])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation4[0], width=bar_width, fc=color_list[3])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation5[0], width=bar_width, fc=color_list[4])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, KGAG[0], width=bar_width, fc=color_list[5])
ax1.grid(axis='y', linestyle='--')
ax1.set_axisbelow(True)
ax1.set_ylabel("Recall@20", fontsize=15, weight='bold')  # Y轴标签
ax1.set_xlabel("Metrics", fontsize=15, weight='bold')  # X轴标签

ax2.tick_params(labelsize=12)

ax2.bar(ax2_x, ablation1[1], width=bar_width, label='KARE w/ GCN', fc=color_list[0])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation2[1], width=bar_width, label='KARE w/o ${L}_{pcl}$', fc=color_list[1])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation3[1], width=bar_width, label='KARE w/ ${L}_{bpr}$', fc=color_list[2])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation4[1], width=bar_width, label='KARE w/o $\\alpha$', fc=color_list[3])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation5[1], width=bar_width, label='KARE w/o ${L}_{ird}$', fc=color_list[4])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, KGAG[1], width=bar_width, label='KARE', fc=color_list[5])
ax2.set_ylabel("NDCG@20", fontsize=15, weight='bold')  # Y轴标签


x1 = [0.3, 0.7]
plt.xticks(x1, name_list)

# plt.legend(fontsize=11, loc='lower left', ncol=1)
plt.legend(fontsize=11, loc='upper right', ncol=1)

if dataset == 'amazonbook':
    plt.title("AmazonBook", fontsize=17, weight='bold')
elif dataset == 'movielens1m_kg':
    plt.title("Movielens-1M", fontsize=17, weight='bold')
elif dataset == 'lastfm_kg':
    plt.title("LastFM", fontsize=17, weight='bold')

for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')

# plt.show()

output_dir = os.path.join(world.PATH_PLOT, 'WORK2', 'RQ2')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, dataset + '_ablation.eps'), dpi=900, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, dataset + '_ablation.png'), dpi=900, bbox_inches='tight')
# plt.close()
# plt.show()