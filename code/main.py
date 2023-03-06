# -*- coding: utf-8 -*-
"""
@Project ：rec_torch 
@File    ：selector.py
@Author  ：Knkiss
@Date    ：2023/2/16 16:44 
"""
import multiprocessing
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

import contrast

import world
import model
import utils


class Metrics(Enum):
    Recall = "recall"
    Precision = "precision"
    NDCG = "NDCG"


class Procedure(Enum):
    Train_Trans = 1
    Train_Rec = 2
    Train_Rec_Contrast = 4
    Test = 3


class Manager:
    def __init__(self):
        self.rec_model = None  # 推荐本体模型
        self.contrast_model = None  # 对比学习模型
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
        self.loop_procedure()
        self.__close()
        utils.mail_on_stop(self.best_result)

    def __prepare_model(self):
        if world.model == 'KGCL':
            self.rec_model = model.KGCL(world.config).to(world.device)
            self.contrast_model = contrast.Contrast(self.rec_model).to(world.device)
            self.procedure = [Procedure.Train_Trans, Procedure.Train_Rec_Contrast, Procedure.Test]
        else:
            self.rec_model = model.Baseline(world.config).to(world.device)
            self.procedure = [Procedure.Train_Rec, Procedure.Test]

    def __prepare_optimizer(self):
        self.optimizer = optim.Adam(self.rec_model.parameters(), lr=world.config['lr'])
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[1500, 2500], gamma=0.2)  # 数据集区别

    def __prepare_tensorboard(self):
        if world.tensorboard:
            self.tensorboard = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d_%Hh%Mm%Ss_"+world.model)))

    def print_rec_module_info(self):
        print("---------------------")
        for i in self.rec_model.modules():
            print(i)
        print("---------------------")
        for i, j in self.rec_model.state_dict().items():
            print(i, ' :', j.shape)
        print("---------------------")

    def loop_procedure(self):
        for self.epoch in range(world.TRAIN_epochs):
            if self.stopping_step == -1:
                print("Best result: "+str(self.best_result))
                break
            for i in self.procedure:
                if i == Procedure.Train_Rec:
                    print("【Main】")
                    self.__procedure_train_Rec()
                elif i == Procedure.Test:
                    self.__procedure_test()
                elif i == Procedure.Train_Trans:
                    print("【Trans】")
                    self.__procedure_train_TransR()
                elif i == Procedure.Train_Rec_Contrast:
                    print("【Main】")
                    self.contrast_model.BPR_train_contrast(self.contrast_model.get_views(),
                                                           self.epoch, self.optimizer, w = self.tensorboard)
                else:
                    raise Exception('不存在的进程类型')
            self.scheduler.step()

    def __procedure_train_TransR(self):
        self.rec_model.train()
        KGLoader = DataLoader(self.rec_model.kg_dataset, batch_size=4096, drop_last=False)
        trans_loss = 0.
        for data in tqdm(KGLoader, total=len(KGLoader), disable=True):
            heads = data[0].to(world.device)
            relations = data[1].to(world.device)
            pos_tails = data[2].to(world.device)
            neg_tails = data[3].to(world.device)
            kg_batch_loss = self.rec_model.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
            trans_loss += kg_batch_loss / len(KGLoader)
            self.optimizer.zero_grad()
            kg_batch_loss.backward()
            self.optimizer.step()
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(f'Loss/Trans', trans_loss, self.epoch)
        print(f"Trans Loss: {trans_loss.cpu().item():.3f}")

    def __procedure_train_Rec(self):
        self.rec_model.train()
        batch_size = world.config['bpr_batch_size']
        UILoader = DataLoader(self.rec_model.ui_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        total_batch = len(UILoader)
        aver_loss = 0.
        for batch_i, train_data in tqdm(enumerate(UILoader), total=len(UILoader), disable=True):
            batch_users = train_data[0].long().to(world.device)
            batch_pos = train_data[1].long().to(world.device)
            batch_neg = train_data[2].long().to(world.device)
            l_main = utils.computeBPR(self.rec_model, batch_users, batch_pos, batch_neg)
            self.optimizer.zero_grad()
            l_main.backward()
            self.optimizer.step()
            aver_loss += l_main.cpu().item()
        aver_loss = aver_loss / (total_batch * batch_size)
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(f'Loss/BPR', aver_loss, self.epoch)
        time_info = utils.timer.dict()
        utils.timer.zero()
        print(f"BPR Loss: {aver_loss:.3f}-{time_info}")

    def __procedure_test(self):
        stop_metric = Metrics.Recall.value
        if self.epoch < world.test_start_epoch and self.epoch % 5 == 0:
            print("【TEST】")
            self.best_result = self.__Test()
            print(self.best_result)
        elif self.epoch > world.test_start_epoch and self.epoch % world.test_verbose == 0:
            print("【TEST】")
            result = self.__Test()
            print(result)
            if result[stop_metric] > self.best_result[stop_metric]:
                self.stopping_step = 0
                self.best_result = result
                print("Find a better model")
            else:
                self.stopping_step += 1
                if self.stopping_step >= world.early_stop_cnt:
                    print(f"early stop triggerd at epoch {self.epoch}")
                    self.stopping_step = -1

    def __Test(self):
        self.rec_model.eval()
        u_batch_size = world.config['test_u_batch_size']
        dataset = self.rec_model.ui_dataset
        testDict: dict = dataset.testDict
        max_K = max(world.topks)
        if world.config['multicore'] == 1:
            pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
        results = {'precision': np.zeros(len(world.topks)),
                   'recall': np.zeros(len(world.topks)),
                   'ndcg': np.zeros(len(world.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
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

            def test_one_batch(X):
                sorted_items = X[0].numpy()
                groundTrue = X[1]
                r = utils.getLabel(groundTrue, sorted_items)
                pre, recall, ndcg = [], [], []
                for k in world.topks:
                    ret = utils.RecallPrecision_ATk(groundTrue, r, k)
                    pre.append(ret['precision'])
                    recall.append(ret['recall'])
                    ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
                return {'recall': np.array(recall),
                        'precision': np.array(pre),
                        'ndcg': np.array(ndcg)}

            if world.config['multicore'] == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            if self.tensorboard is not None:
                self.tensorboard.add_scalars(f'Test/Recall@{world.topks}',
                                             {str(world.topks[i]): results['recall'][i] for i in
                                              range(len(world.topks))}, self.epoch)
                self.tensorboard.add_scalars(f'Test/Precision@{world.topks}',
                                             {str(world.topks[i]): results['precision'][i] for i in
                                              range(len(world.topks))}, self.epoch)
                self.tensorboard.add_scalars(f'Test/NDCG@{world.topks}',
                                             {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))},
                                             self.epoch)
            if world.config['multicore'] == 1:
                pool.close()
            return results

    def __close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()


if __name__ == '__main__':
    Manager()
