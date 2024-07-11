# -*- coding: utf-8 -*-
"""
@Project ：rec_torch
@File    ：main_ablation.py
@Author  ：Knkiss
@Date    ：2023/8/23 12:00
"""
from main import Manager
import world


class Ablation:
    def __init__(self, ablations, datasets, postfix):
        # 展示信息全关
        world.tqdm_enable = False
        world.epoch_result_show_enable = False
        world.model_info_show_enable = False
        world.dataset_info_show_enable = False

        # 保存结果全开
        world.predict_list_enable = True
        world.early_stop_enable = True

        self.ablation_list = ablations
        self.dataset_list = datasets
        self.ablation_experiments(postfix)

    def ablation_experiments(self, postfix):
        for i in self.dataset_list:
            for j in self.ablation_list:
                world.dataset = i
                world.hyper_WORK2_ablation_model = j
                world.sys_ablation_name = str(j) + str(postfix)
                world.sys_root_model = False
                Manager()


if __name__ == '__main__':
    postfix = 'ablation'
    ablation_list = [1, 2, 3, 4]
    dataset_list = ['lastfm_kg', 'movielens1m_kg', 'amazonbook']
    Ablation(ablation_list, dataset_list, postfix)
