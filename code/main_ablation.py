# -*- coding: utf-8 -*-
"""
@Project ：rec_torch
@File    ：main_ablation.py
@Author  ：Knkiss
@Date    ：2023/8/23 12:00
"""
import main
import world


class Ablation:
    def __init__(self, ablations, datasets):
        self.results = {}
        self.best_result = None
        world.predict_list_enable = True
        world.early_stop_enable = True
        self.ablation_list = ablations
        self.dataset_list = datasets
        self.ablation_experiments()

    def ablation_experiments(self):
        for i in self.dataset_list:
            for j in self.ablation_list:
                world.dataset = i
                # TODO 使用引用变量类型令此参数作为外部输入
                world.hyper_KGCL_my_ablated_model = j
                world.sys_ablation_name = str(j)
                main.Manager()


if __name__ == '__main__':
    ablation_list = [0, 1, 2, 3]
    dataset_list = ['amazonbook', 'movielens1m_kg', 'lastfm_kg']
    # UI数据集: 'citeulikea', 'lastfm', 'movielens1m', 'yelp2018'
    # KG数据集: 'amazonbook', 'yelp2018_kg', 'bookcrossing', 'movielens1m_kg', 'lastfm_kg', 'lastfm_wxkg'

    Ablation(ablation_list, dataset_list)
