# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：main.py
@Author  ：Knkiss
@Date    ：2023/2/16 16:44 
"""
import itertools
import sys
import time
from enum import Enum
from os.path import join

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from train.losses import Loss
from train import metrics, utils
import world
from train.metrics import Metrics


class Procedure(Enum):
    Train_Trans = 1
    Train_Rec = 2
    Test = 3


class Manager:
    def __init__(self):
        self.rec_model: model.AbstractRecModel  # 推荐模型
        self.optimizer = None  # 优化器，计算梯度
        self.scheduler = None  # 学习率调整器
        self.tensorboard = None  # 可视化损失和结果

        self.epoch = 0  # 迭代计数器
        self.stopping_step = 0  # 累加停止计数器，等于-1时训练停止
        self.best_result = {}  # 性能统计

        self.procedure = []

        utils.set_seed(world.seed)
        self.__prepare_model()
        self.__prepare_optimizer()
        self.__prepare_tensorboard()
        self.print_rec_module_info()
        time_start = time.time()
        self.__loop_procedure()
        time_end = time.time()
        self.__close()
        utils.mail_on_stop(self.best_result)
        print("Time Spend: ", time_end - time_start, "s")

    def __prepare_model(self):
        self.procedure = [Procedure.Train_Rec, Procedure.Test]
        self.rec_model = model.get_model_by_name(world.model)
        self.rec_model = self.rec_model.to(world.device)

        # DIFF 性能提升 去掉Trans的计算
        if world.model == 'KGCL':
            self.procedure = [Procedure.Train_Trans, Procedure.Train_Rec, Procedure.Test]

    def __prepare_optimizer(self):
        self.optimizer = optim.Adam(self.rec_model.parameters(), lr=world.learning_rate)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[1500, 2500], gamma=0.2)  # 数据集区别

    def __prepare_tensorboard(self):
        if world.tensorboard_enable:
            self.tensorboard = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d_%Hh%Mm%Ss_" + world.model)))
            world.tensorboard_instance = self.tensorboard

    def print_rec_module_info(self):
        print("--------------------- Modules  ---------------------")
        print(self.rec_model)
        print("----------------------------------------------------")
        for i, j in self.rec_model.state_dict().items():
            print(i, ' :', j.shape)
        print("----------------------------------------------------")

    def __loop_procedure(self):
        for self.epoch in range(0, world.TRAIN_epochs):
            world.epoch = self.epoch
            if self.stopping_step == -1 or self.epoch == world.TRAIN_epochs - 1:
                print("Best result: " + str(self.best_result))
                break
            for i in self.procedure:
                if i == Procedure.Train_Rec:
                    self.__procedure_train_Rec()
                elif i == Procedure.Test:
                    self.__procedure_test()
                elif i == Procedure.Train_Trans:
                    self.__procedure_train_TransR()
                else:
                    raise Exception('不存在的进程类型')
            self.scheduler.step()

    def __procedure_train_TransR(self):
        self.rec_model.train()
        KGLoader = DataLoader(self.rec_model.kg_dataset, batch_size=4096, drop_last=False)
        trans_loss = 0.
        with tqdm(KGLoader, file=sys.stdout, total=len(KGLoader),
                  desc='Trans Epoch ' + str(world.epoch).zfill(3), disable=not world.tqdm_enable) as t:
            for data in t:
                heads = data[0].to(world.device)
                relations = data[1].to(world.device)
                pos_tails = data[2].to(world.device)
                neg_tails = data[3].to(world.device)
                kg_batch_loss = self.rec_model.calculate_loss_transE(heads, relations, pos_tails, neg_tails)
                self.optimizer.zero_grad()
                kg_batch_loss.backward()
                self.optimizer.step()
                trans_loss += kg_batch_loss.cpu().item()
            t.close()
        if world.tensorboard_enable:
            self.tensorboard.add_scalar(f'Loss/Trans', trans_loss / len(KGLoader), self.epoch)

    def __procedure_train_Rec(self):
        self.rec_model.train()
        batch_size = world.train_batch_size
        UILoader = DataLoader(self.rec_model.ui_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
        aver_loss = {Loss.BPR.value: 0.}
        for key in Loss:
            aver_loss[key.value] = 0.
        self.rec_model.prepare_each_epoch()
        with tqdm(enumerate(UILoader), file=sys.stdout, total=len(UILoader),
                  desc='[' + world.model + '] Epoch ' + str(self.epoch).zfill(3), disable=not world.tqdm_enable) as t:
            for _, train_data in t:
                batch_users = train_data[0].long().to(world.device)
                batch_pos = train_data[1].long().to(world.device)
                batch_neg = train_data[2].long().to(world.device)
                losses = self.rec_model.calculate_loss(batch_users, batch_pos, batch_neg)
                loss_all = torch.stack(list(losses.values())).sum(dim=0)
                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()
                for key in losses.keys():
                    aver_loss[key] += losses[key].cpu().item()
                t.set_postfix(
                    loss_bpr=aver_loss[Loss.BPR.value] / (batch_size * len(UILoader)),
                    loss_ssl=aver_loss[Loss.SSL.value] / (batch_size * len(UILoader))
                )
        if world.tensorboard_enable:
            self.tensorboard.add_scalar(f'Loss/BPR', aver_loss[Loss.BPR.value] / (batch_size * len(UILoader)),
                                        self.epoch)
            self.tensorboard.add_scalar(f'Loss/SSL', aver_loss[Loss.SSL.value] / (batch_size * len(UILoader)),
                                        self.epoch)

    def __procedure_test(self):
        stop_metric = world.early_stop_metric
        if self.epoch == 0 or (self.epoch < world.test_start_epoch and self.epoch % 5 == 0):
            self.best_result = self.__Test()
            print('\033[0;31m' + str(self.best_result) + '\033[0m')
        elif self.epoch >= world.test_start_epoch and self.epoch % world.test_verbose_epoch == 0:
            result = self.__Test()
            if len(world.topKs) == 1:
                now = result[stop_metric]
                best = self.best_result[stop_metric]
            else:
                now = result[stop_metric][len(world.topKs) - 1]
                best = self.best_result[stop_metric][len(world.topKs) - 1]
            if now > best:
                self.stopping_step = 0
                self.best_result = result
                print('\033[0;31m' + str(result) + ' Find a better model' + '\033[0m')
                if world.pretrain_output_enable:
                    output = world.PRETRAIN_PATH + '/' + world.dataset + '_' + world.model + '.pretrain'
                    torch.save(self.rec_model.state_dict(), output)
            elif world.early_stop_enable:
                print('\033[0;32m' + str(result) + '\033[0m')
                self.stopping_step += 1
                if self.stopping_step >= world.early_stop_epoch_cnt:
                    print(f"early stop triggerd at epoch {self.epoch}")
                    self.stopping_step = -1
            else:
                print('\033[0;32m' + str(result) + '\033[0m')

    def __Test(self):
        self.rec_model.eval()
        u_batch_size = world.test_u_batch_size
        dataset = self.rec_model.ui_dataset
        testDict: dict = dataset.testDict
        max_K = max(world.topKs)
        results = {}
        for i in world.metrics:
            results[i] = np.zeros(len(world.topKs))
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                pass
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = self.rec_model.getUsersRating(batch_users_gpu)
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)

            def test_one_batch(batch):
                sorted_items = batch[0].numpy()
                label = batch[1]
                r = utils.getLabel(label, sorted_items)
                result_list = {}
                for l in world.metrics:
                    result_list[l] = []
                for k in world.topKs:
                    for j in world.metrics:
                        if j == Metrics.Precision.value:
                            result_list[j].append(metrics.Precision_topK(r, k))
                        elif j == Metrics.Recall.value:
                            result_list[j].append(metrics.Recall_topK(label, r, k))
                        elif j == Metrics.NDCG.value:
                            result_list[j].append(metrics.NDCG_topK(label, r, k))
                        elif j == Metrics.MRR.value:
                            result_list[j].append(metrics.MRR_topK(r, k))
                return result_list

            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
            for result in pre_results:
                for i in world.metrics:
                    results[i] += result[i]
            for i in world.metrics:
                results[i] /= float(len(users))
            if self.tensorboard is not None:
                for metric in world.metrics:
                    self.tensorboard.add_scalars(f'Test/' + metric + '@{world.topKs}',
                                                 {str(world.topKs[i]): results[metric][i] for i in
                                                  range(len(world.topKs))}, self.epoch)
            return results

    def __close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()


class Search:
    def __init__(self):
        self.result = []
        self.best_result = None
        self.parameter_table = []
        self.set_parameters_table()
        print(self.parameter_table)
        if input("参数设置列表如上，是否继续进行[y/n]:").lower() != 'y':
            exit(1)
        print("--------------------- Searches ---------------------")
        self.search_parameter()
        self.print_result(self.result)

    def search_parameter(self):
        for parameter in itertools.product(*self.parameter_table):
            parameter_dict = self.set_parameters(parameter)
            result = Manager().best_result
            self.result.append(dict(result, **parameter_dict))
            world.epoch = 0
            world.root_model = False

    def print_result(self, result):
        if result is self.result:
            print("--------------------- Searches ---------------------")
        if isinstance(result, list):
            for i in result:
                self.print_result(i)
        elif isinstance(result, dict):
            if self.best_result is None \
                    or self.best_result[world.early_stop_metric][-1] < result[world.early_stop_metric][-1]:
                self.best_result = result
            print(result)
        if result is self.result:
            print("--------------------- Searches End ------------------")
            print("Best result: " + str(self.best_result))

    # Need Change
    @staticmethod
    def set_parameters(parameters):
        para_dict = {}
        world.CCL_Margin = parameters[0]
        para_dict['CCL_Margin'] = parameters[0]
        return para_dict

    # Need Change
    def set_parameters_table(self):
        self.parameter_table = [[0.1, 0.3, 0.5, 0.7, 0.9]]
        # for i in range(1, 6):
        #     self.parameter_table[0].append(i/10)


if __name__ == '__main__':
    if world.searcher:
        Search()
    else:
        Manager()
