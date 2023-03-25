# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：main.py
@Author  ：Knkiss
@Date    ：2023/2/16 16:44 
"""
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
import utils
import world


class Loss(Enum):
    BPR = 'BPR'
    SSL = 'SSL'
    TransE = 'TransE'
    Regulation = 'reg'


class Metrics(Enum):
    Recall = "Recall"
    Precision = "Precision"
    NDCG = "NDCG"


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
        self.best_result = 0.  # 性能统计

        self.procedure = []

        utils.set_seed(world.seed)
        self.__prepare_model()
        self.__prepare_optimizer()
        self.__prepare_tensorboard()
        self.print_rec_module_info()
        self.__loop_procedure()
        self.__close()
        utils.mail_on_stop(self.best_result)

    def __prepare_model(self):
        self.procedure = [Procedure.Train_Rec, Procedure.Test]
        if world.model == 'KGCL':
            self.rec_model = model.KGCL()
            if not world.remove_Trans:
                self.procedure = [Procedure.Train_Trans, Procedure.Train_Rec, Procedure.Test]
        elif world.model == 'QKV':
            self.rec_model = model.QKV()
        elif world.model == 'SGL' or world.model == 'GraphCL':
            self.rec_model = model.SGL()
        else:
            self.rec_model = model.Baseline()
        self.rec_model = self.rec_model.to(world.device)

    def __prepare_optimizer(self):
        self.optimizer = optim.Adam(self.rec_model.parameters(), lr=world.config['lr'])
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[1500, 2500], gamma=0.2)  # 数据集区别

    def __prepare_tensorboard(self):
        if world.tensorboard_enable:
            self.tensorboard = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d_%Hh%Mm%Ss_" + world.model)))
            world.tensorboard_instance = self.tensorboard

    def print_rec_module_info(self):
        print("---------------------")
        for i in self.rec_model.modules():
            print(i)
        print("---------------------")
        for i, j in self.rec_model.state_dict().items():
            print(i, ' :', j.shape)
        print("---------------------")

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
        batch_size = world.config['train_batch_size']
        UILoader = DataLoader(self.rec_model.ui_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
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
        stop_metric = Metrics.Recall.value
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
                    output = world.pretrain_folder + world.dataset + '_' + world.model + '.pretrain'
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
        u_batch_size = world.config['test_u_batch_size']
        dataset = self.rec_model.ui_dataset
        testDict: dict = dataset.testDict
        max_K = max(world.topKs)
        results = {Metrics.Precision.value: np.zeros(len(world.topKs)),
                   Metrics.NDCG.value: np.zeros(len(world.topKs)),
                   Metrics.Recall.value: np.zeros(len(world.topKs))}
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
                pre, recall, ndcg = [], [], []
                for k in world.topKs:
                    ret = utils.RecallPrecision_ATk(label, r, k)
                    pre.append(ret[Metrics.Precision.value])
                    recall.append(ret[Metrics.Recall.value])
                    ndcg.append(utils.NDCGatK_r(label, r, k))
                return {Metrics.Recall.value: np.array(recall),
                        Metrics.Precision.value: np.array(pre),
                        Metrics.NDCG.value: np.array(ndcg)}

            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
            for result in pre_results:
                results[Metrics.Recall.value] += result[Metrics.Recall.value]
                results[Metrics.Precision.value] += result[Metrics.Precision.value]
                results[Metrics.NDCG.value] += result[Metrics.NDCG.value]
            results[Metrics.Recall.value] /= float(len(users))
            results[Metrics.Precision.value] /= float(len(users))
            results[Metrics.NDCG.value] /= float(len(users))
            if self.tensorboard is not None:
                self.tensorboard.add_scalars(f'Test/Recall@{world.topKs}',
                                             {str(world.topKs[i]): results[Metrics.Recall.value][i] for i in
                                              range(len(world.topKs))}, self.epoch)
                self.tensorboard.add_scalars(f'Test/Precision@{world.topKs}',
                                             {str(world.topKs[i]): results[Metrics.Precision.value][i] for i in
                                              range(len(world.topKs))}, self.epoch)
                self.tensorboard.add_scalars(f'Test/NDCG@{world.topKs}',
                                             {str(world.topKs[i]): results[Metrics.NDCG.value][i] for i in
                                              range(len(world.topKs))},
                                             self.epoch)
            return results

    def __close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()


if __name__ == '__main__':
    Manager()
