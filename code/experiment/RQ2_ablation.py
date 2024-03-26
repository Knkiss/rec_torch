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
    KGAG = [0.1519, 0.1190]
    ablation1 = [0.1505, 0.1154]
    KGIC_WO_ALPHA = [0.1503, 0.1162]
    ablation_add1 = [0.1508, 0.1171]
    ablation2 = [0.15168642, 0.1183496]
    ablation3 = [0.15160535, 0.118323]
    ablation4 = [0.15163664, 0.11840603]
elif dataset == 'movielens1m_kg':
    KGAG = [0.3384, 0.5567]
    ablation1 = [0.3185, 0.5404]
    KGIC_WO_ALPHA = [0.3133, 0.5390]
    ablation_add1 = [0.3269, 0.5495]
    ablation2 = [0.33826291, 0.55063179]
    ablation3 = [0.33817706, 0.55238419]
    ablation4 = [0.33815515, 0.55150593]
elif dataset == 'lastfm_kg':
    KGAG = [0.3870, 0.3436]
    ablation1 = [0.3796, 0.3307]
    KGIC_WO_ALPHA = [0.3633, 0.3184]
    ablation_add1 = [0.3817, 0.3380]
    ablation2 = [0.3851012, 0.34337522]
    ablation3 = [0.38410771, 0.33770216]
    ablation4 = [0.38608042, 0.34324933]


bar_width = 0.045
bar_and_space_width = 0.05
ax1_x = [0.175]
ax2_x = [0.575]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlim([0.1, 0.9])

if dataset == 'amazonbook':
    ax1.set_ylim([0.15, 0.152])
    ax1.set_yticks([0.150, 0.1505, 0.151, 0.1515, 0.152])
    ax2.set_ylim([0.112, 0.127])
    ax2.set_yticks([0.112, 0.115, 0.118, 0.121, 0.124, 0.127])
elif dataset == 'movielens1m_kg':
    ax1.set_ylim([0.310, 0.34])
    ax1.set_yticks([0.310, 0.315, 0.32, 0.325, 0.33, 0.335, 0.340])
    ax2.set_ylim([0.53, 0.59])
    ax2.set_yticks([0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
elif dataset == 'lastfm_kg':
    ax1.set_ylim([0.36, 0.39])
    ax1.set_yticks([0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39])
    ax2.set_ylim([0.31, 0.38])
    ax2.set_yticks([0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38])

ax1.tick_params(labelsize=12)
ax1.bar(ax1_x, ablation1[0], width=bar_width,fc=color_list[-1])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, KGIC_WO_ALPHA[0], width=bar_width,fc=color_list[0])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation_add1[0], width=bar_width,fc=color_list[1])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation2[0], width=bar_width, fc=color_list[2])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation3[0], width=bar_width, fc=color_list[3])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, ablation4[0], width=bar_width, fc=color_list[4])
ax1_x[0] += bar_and_space_width
ax1.bar(ax1_x, KGAG[0], width=bar_width, fc=color_list[5])
ax1.grid(axis='y', linestyle='--')
ax1.set_axisbelow(True)
ax1.set_ylabel("Recall@20", fontsize=15, weight='bold')  # Y轴标签
ax1.set_xlabel("Metrics", fontsize=15, weight='bold')  # X轴标签

ax2.tick_params(labelsize=12)
ax2.bar(ax2_x, ablation1[1], width=bar_width,  label='KGIC w/o KG', fc=color_list[-1])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, KGIC_WO_ALPHA[1], width=bar_width,  label='KGIC w/o $\\alpha$', fc=color_list[0])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation_add1[1], width=bar_width,  label='KGIC w/o $p_{ui}$', fc=color_list[1])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation2[1], width=bar_width, label='KGIC w/ $p_u$', fc=color_list[2])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation3[1], width=bar_width, label='KGIC w/ $p_i$', fc=color_list[3])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, ablation4[1], width=bar_width, label='KGIC w/ $p_{ui}$ concat', fc=color_list[4])
ax2_x[0] += bar_and_space_width
ax2.bar(ax2_x, KGAG[1], width=bar_width, label='KGIC', fc=color_list[5])
ax2.set_ylabel("NDCG@20", fontsize=15, weight='bold')  # Y轴标签


x1 = [0.3, 0.7]
plt.xticks(x1, name_list)

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

output_dir = os.path.join(world.PATH_PLOT, 'KGAG', 'RQ2')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# plt.savefig(os.path.join(output_dir, dataset + '_ablation.eps'), dpi=900, bbox_inches='tight')
# plt.savefig(os.path.join(output_dir, dataset + '_ablation.png'), dpi=900, bbox_inches='tight')
# plt.close()
# plt.show()