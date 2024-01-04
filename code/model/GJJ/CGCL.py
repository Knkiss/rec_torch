import torch
from recbole.model.init import xavier_normal_initialization
from torch import nn

import model
from train import losses

group_number = 2


class CGCL(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.factual_embs = []
        self.counterfactual_embs_u = []
        self.counterfactual_embs_i = []

        self.A_fold_hat_u = None
        self.A_fold_hat_i = None
        self.group = group_number

        # 子模型网络
        self.base_model = model.LightGCN()

        # 用于群体分组
        self.group_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_dim, group_number, bias=True),
        )

        self.apply(xavier_normal_initialization)

    def _split_A_hat_group(self, X, group_embedding):
        group_embedding = group_embedding.T
        A_fold_hat_group = []
        A_fold_hat_group_filter = []

        # Graph_hat = utils.convert_sp_mat_to_sp_tensor(X).to(world.device)
        Graph_hat = X

        # k groups
        for k in range(0, self.group):
            A_fold_item_filter = []

            a1 = Graph_hat * (group_embedding[k])
            a2 = group_embedding[k].unsqueeze(1)
            A_fold_hat_item = a1 * a2

            item_filter = torch.sum(A_fold_hat_item, dim=1)

            item_filter_coo = item_filter.to_dense()
            ones_coo = torch.ones_like(item_filter_coo)
            zeros_coo = torch.zeros_like(item_filter_coo)
            item_filter = torch.where(item_filter_coo > 0., ones_coo, zeros_coo)
            A_fold_item_filter.append(item_filter)

            A_fold_item = torch.concat(A_fold_item_filter, dim=0)
            A_fold_hat_group_filter.append(A_fold_item)
            A_fold_hat_group.append(A_fold_hat_item)

        return A_fold_hat_group, A_fold_hat_group_filter

    # def prepare_each_epoch(self):
        # self.factual_embs = []
        # self.counterfactual_embs_u = []
        # self.counterfactual_embs_i = []
        #
        # # MODULE 第0层结果
        # embs_0 = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        # self.factual_embs.append(embs_0)
        # for i in range(self.group):
        #     self.counterfactual_embs_u.append([embs_0])
        #     self.counterfactual_embs_i.append([embs_0])
        #
        # # MODULE 第1层结果
        # embs_1 = torch.cat(self.base_model(self.embedding_user.weight, self.embedding_item.weight,
        #                                    self.Graph, n_layers=1, output_one_layer=True), dim=0)
        # self.factual_embs.append(embs_1)
        # for i in range(self.group):
        #     self.counterfactual_embs_u[i].append(embs_1)
        #     self.counterfactual_embs_i[i].append(embs_1)

    def calculate_embedding(self):
        self.factual_embs = []
        self.counterfactual_embs_u = []
        self.counterfactual_embs_i = []

        # MODULE 第0层结果
        embs_0 = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        self.factual_embs.append(embs_0)
        for i in range(self.group):
            self.counterfactual_embs_u.append([embs_0])
            self.counterfactual_embs_i.append([embs_0])

        # MODULE 第1层结果
        embs_1 = torch.cat(self.base_model(self.embedding_user.weight, self.embedding_item.weight,
                                           self.Graph, n_layers=1, output_one_layer=True), dim=0)
        self.factual_embs.append(embs_1)
        for i in range(self.group):
            self.counterfactual_embs_u[i].append(embs_1)
            self.counterfactual_embs_i[i].append(embs_1)

        # MODULE 根据第一层结果分割组
        group_emb = self.group_mlp(self.factual_embs[1])
        a_top, a_top_idx = torch.topk(group_emb, 1, largest=False)
        group_emb = torch.eq(group_emb, a_top).to(torch.float32)

        # MODULE 根据分组结果生成掩膜
        u_group_embeddings, i_group_embeddings = torch.split(group_emb, [self.n_users, self.n_items], dim=0)
        user_group_embeddings = torch.cat((u_group_embeddings, torch.ones_like(i_group_embeddings)), dim=0)
        item_group_embeddings = torch.cat((torch.ones_like(u_group_embeddings), i_group_embeddings), dim=0)

        # MODULE 根据掩膜获得分组矩阵
        self.A_fold_hat_u, A_fold_hat_u_filter = self._split_A_hat_group(self.Graph, user_group_embeddings)
        self.A_fold_hat_i, A_fold_hat_i_filter = self._split_A_hat_group(self.Graph, item_group_embeddings)

        # MODULE 事实世界 第2、3层结果
        all_users, all_items = torch.split(self.factual_embs[1], [self.n_users, self.n_items], dim=0)
        self.factual_embs.append(torch.cat(self.base_model(all_users, all_items,
                                                           self.Graph, n_layers=1, output_one_layer=True), dim=0))
        all_users, all_items = torch.split(self.factual_embs[2], [self.n_users, self.n_items], dim=0)
        self.factual_embs.append(torch.cat(self.base_model(all_users, all_items,
                                                           self.Graph, n_layers=1, output_one_layer=True), dim=0))

        self.factual_embs = torch.mean(torch.stack(self.factual_embs, dim=0), dim=0)

        # MODULE 反事实世界 第2、3层结果，用户和物品均有分组
        for g in range(self.group):
            all_users, all_items = torch.split(self.counterfactual_embs_u[g][1], [self.n_users, self.n_items], dim=0)
            self.counterfactual_embs_u[g].append(
                torch.cat(self.base_model(all_users, all_items,
                                          self.A_fold_hat_u[g].coalesce(), n_layers=1, output_one_layer=True), dim=0))
            all_users, all_items = torch.split(self.counterfactual_embs_u[g][2], [self.n_users, self.n_items], dim=0)
            self.counterfactual_embs_u[g].append(
                torch.cat(self.base_model(all_users, all_items,
                                          self.A_fold_hat_u[g].coalesce(), n_layers=1, output_one_layer=True), dim=0))

            # 物品分组计算emb，使用物品图
            all_users, all_items = torch.split(self.counterfactual_embs_i[g][1], [self.n_users, self.n_items], dim=0)
            self.counterfactual_embs_i[g].append(
                torch.cat(self.base_model(all_users, all_items,
                                          self.A_fold_hat_i[g].coalesce(), n_layers=1, output_one_layer=True), dim=0))
            all_users, all_items = torch.split(self.counterfactual_embs_i[g][2], [self.n_users, self.n_items], dim=0)
            self.counterfactual_embs_i[g].append(
                torch.cat(self.base_model(all_users, all_items,
                                          self.A_fold_hat_i[g].coalesce(), n_layers=1, output_one_layer=True), dim=0))

        for g in range(self.group):
            self.counterfactual_embs_u[g] = torch.mean(torch.stack(self.counterfactual_embs_u[g], dim=0), dim=0)
            self.counterfactual_embs_i[g] = torch.mean(torch.stack(self.counterfactual_embs_i[g], dim=0), dim=0)

        # MODULE 从不同组中取相应的embs
        u_emb = self.factual_embs[0]
        i_emb = self.factual_embs[1]

        u_sub_emb_0, i_sub_emb_1 = torch.split(self.counterfactual_embs_u[0], [self.n_users, self.n_items], 0)
        u_sub_emb_0, i_sub_emb_2 = torch.split(self.counterfactual_embs_u[1], [self.n_users, self.n_items], 0)
        u_sub_emb_1, i_sub_emb_0 = torch.split(self.counterfactual_embs_i[0], [self.n_users, self.n_items], 0)
        u_sub_emb_2, i_sub_emb_0 = torch.split(self.counterfactual_embs_i[1], [self.n_users, self.n_items], 0)

        # MODULE 频繁嵌入和不频繁嵌入计算、融合
        frequent_embs_u = 0.5 * (u_sub_emb_1 + u_sub_emb_2)
        frequent_embs_i = 0.5 * (i_sub_emb_1 + i_sub_emb_2)

        infrequent_embs_u = (2 * u_emb - u_sub_emb_1 - u_sub_emb_2)
        infrequent_embs_i = (2 * i_emb - i_sub_emb_1 - i_sub_emb_2)

        res_embs_u = torch.concat([frequent_embs_u, infrequent_embs_u], 1)
        res_embs_i = torch.concat([frequent_embs_i, infrequent_embs_i], 1)

        return res_embs_u, res_embs_i

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                    pos, neg)
        return loss
