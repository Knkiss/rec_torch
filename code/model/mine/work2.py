import torch
import numpy as np
import torch.nn as nn
from torch.distributions import RelaxedBernoulli

import model
import world
from train import losses, utils
from scipy.sparse import csr_matrix
from torch.utils.data.dataloader import default_collate


class MatrixRebuild(nn.Module):
    """
    根据输入的emb，得到一个生成的graph_random，其中包含一定的随机噪声
    """

    def __init__(self, embedding_dim, n_users, n_items):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_users = n_users
        self.n_items = n_items

        self.mlp_edge_model = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        ).to(world.device)

        self.graph_random = None  # 每个epoch随机产生的额外graph

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def prepare_each_epoch(self):
        # 交互图的随机噪声，源代码是100000个
        number = int(100000)
        rdmUsrs = torch.randint(self.n_users, [number])  # ancs
        rdmItms1 = torch.randint_like(rdmUsrs, self.n_items)
        new_idxs = default_collate([rdmUsrs, rdmItms1])
        new_vals = torch.tensor([0.05] * number)
        shape = torch.Size((self.n_items + self.n_users, self.n_items + self.n_users))
        self.graph_random = torch.sparse_coo_tensor(new_idxs, new_vals, shape).to(world.device)

    def forward(self, all_emb, graph):
        edge_index = graph._indices()
        src, dst = edge_index[0], edge_index[1]
        emb_src = all_emb[src]
        emb_dst = all_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)

        edge_logits = self.mlp_edge_model(edge_emb)  # u、i的emb通过神经网络计算得到edge_logits
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)  # [1-bias, bias]
        gate_inputs = torch.log(eps) - torch.log(1 - eps)  # 均匀噪声转换为gumble噪声
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_wight = torch.sigmoid(gate_inputs).squeeze()  # 整理到0和1之间
        mat = self.__build_prob_neighbourhood(edge_wight, 0.9)
        graph = torch.sparse_coo_tensor(edge_index, mat, self.graph_random.shape).to(world.device)
        return (graph + self.graph_random).coalesce()

    @staticmethod
    def __build_prob_neighbourhood(edge_wight, temperature):
        attention = torch.clamp(edge_wight, 0.01, 0.99)
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()
        eps = 0.0
        mask = (weighted_adjacency_matrix > eps).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
        return weighted_adjacency_matrix


class MatrixResample():
    """
    根据输入的预训练emb，采样得到一个包含uuii的graph，不包含噪声
    """

    def __init__(self, ui_dataset, n_users, n_items):
        super().__init__()
        self.ui_dataset = ui_dataset
        self.n_users = n_users
        self.n_items = n_items

        emb = torch.load(world.PATH_PRETRAIN + '/' + world.dataset + '_' + world.pretrain_input + '.pretrain')
        self.pretrained_embedding_user = torch.nn.Embedding.from_pretrained(emb['embedding_user.weight']).weight
        self.pretrained_embedding_item = torch.nn.Embedding.from_pretrained(emb['embedding_item.weight']).weight

        self.sample_batch_size = 100
        self.distill_userK = 3  # <= 50
        self.distill_itemK = 3  # <= 50
        self.distill_uuK = 3
        self.distill_iiK = 3
        self.distill_thres = 0.5
        self.uuii_thres = -1.0
        self.f = nn.Sigmoid()

        sampled_ui = self.sample_ui_from_pretrained()
        self.reset_ui_dataset(sampled_ui)
        self.Graph = self.ui_dataset.getSparseGraph(include_uuii=True, regenerate_not_save=True)

    def __get_pretrained_ui_rating(self, users):
        all_users = self.pretrained_embedding_user
        all_items = self.pretrained_embedding_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def __get_pretrained_uu_rating(self, users):
        all_users = self.pretrained_embedding_user
        users_emb = all_users[users.long()]
        return self.f(torch.matmul(users_emb, all_users.t()))

    def __get_pretrained_iu_rating(self, items):
        all_users = self.pretrained_embedding_user
        all_items = self.pretrained_embedding_item
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating

    def __get_pretrained_ii_rating(self, items):
        all_items = self.pretrained_embedding_item
        items_emb = all_items[items.long()]
        return self.f(torch.matmul(items_emb, all_items.t()))

    def sample_ui_from_pretrained(self):
        distill_user_row = []
        distill_item_col = []
        distill_value = []

        distill_uu_row = []
        distill_uu_col = []
        distill_uu_value = []

        distill_ii_row = []
        distill_ii_col = []
        distill_ii_value = []

        u_batch_size = self.sample_batch_size

        if self.distill_userK > 0:
            with torch.no_grad():
                users = list(set(self.ui_dataset.trainUser))
                try:
                    assert u_batch_size <= len(users) / 10
                except AssertionError:
                    raise ValueError(
                        f"sample_batch_size is too big for this dataset, try a small one {len(users) // 10}")

                for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                    batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)

                    rating_pred = self.__get_pretrained_ui_rating(batch_users_gpu)
                    uu_pred = self.__get_pretrained_uu_rating(batch_users_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    uu_pred = uu_pred.cpu().data.numpy().copy()
                    uu_ind = np.argpartition(uu_pred, -50)[:, -50:]
                    uu_arr_ind = uu_pred[np.arange(len(uu_pred))[:, None], uu_ind]
                    uu_arr_ind_argsort = np.argsort(uu_arr_ind)[np.arange(len(uu_pred)), ::-1]
                    uu_batch_pred_list = uu_ind[np.arange(len(uu_pred))[:, None], uu_arr_ind_argsort]

                    partial_batch_pred_list = batch_pred_list[:, :self.distill_userK]
                    uu_partial_batch_pred_list = uu_batch_pred_list[:, :self.distill_uuK]

                    for batch_i in range(partial_batch_pred_list.shape[0]):
                        uid = batch_users[batch_i]
                        user_pred = partial_batch_pred_list[batch_i]
                        uu_user_pred = uu_partial_batch_pred_list[batch_i]
                        for eachpred in user_pred:
                            distill_user_row.append(uid)
                            distill_item_col.append(eachpred)
                            pred_val = rating_pred[batch_i, eachpred]
                            if self.distill_thres > 0:
                                if pred_val > self.distill_thres:
                                    distill_value.append(1)
                                else:
                                    distill_value.append(0)
                            else:
                                distill_value.append(pred_val)

                        for eachpred in uu_user_pred:
                            distill_uu_row.append(uid)
                            distill_uu_col.append(eachpred)
                            distill_uu_row.append(eachpred)
                            distill_uu_col.append(uid)
                            pred_val = uu_pred[batch_i, eachpred]
                            if self.uuii_thres > 0:
                                if pred_val > self.uuii_thres:
                                    distill_uu_value.append(1)
                                    distill_uu_value.append(1)
                                else:
                                    distill_uu_value.append(0)
                                    distill_uu_value.append(0)
                            else:
                                distill_uu_value.append(pred_val)
                                distill_uu_value.append(pred_val)

        if self.distill_itemK > 0:
            with torch.no_grad():
                items = [i for i in range(self.n_items)]
                total_batch = len(items) // u_batch_size + 1
                for batch_items in utils.minibatch(items, batch_size=u_batch_size):
                    batch_items_gpu = torch.Tensor(batch_items).long().to(world.device)

                    rating_pred = self.__get_pretrained_iu_rating(batch_items_gpu)
                    ii_pred = self.__get_pretrained_ii_rating(batch_items_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    ii_pred = ii_pred.cpu().data.numpy().copy()
                    ii_ind = np.argpartition(ii_pred, -50)[:, -50:]
                    ii_arr_ind = ii_pred[np.arange(len(ii_pred))[:, None], ii_ind]
                    ii_arr_ind_argsort = np.argsort(ii_arr_ind)[np.arange(len(ii_pred)), ::-1]
                    ii_batch_pred_list = ii_ind[np.arange(len(ii_pred))[:, None], ii_arr_ind_argsort]

                    partial_batch_pred_list = batch_pred_list[:, :self.distill_itemK]
                    ii_partial_batch_pred_list = ii_batch_pred_list[:, :self.distill_iiK]
                    for batch_i in range(partial_batch_pred_list.shape[0]):
                        iid = batch_items[batch_i]
                        item_pred = partial_batch_pred_list[batch_i]
                        ii_item_pred = ii_partial_batch_pred_list[batch_i]
                        for eachpred in item_pred:
                            distill_user_row.append(eachpred)
                            distill_item_col.append(iid)
                            pred_val = rating_pred[batch_i, eachpred]
                            if self.distill_thres > 0:
                                if pred_val > self.distill_thres:
                                    distill_value.append(1)
                                else:
                                    distill_value.append(0)
                            else:
                                distill_value.append(pred_val)

                        for eachpred in ii_item_pred:
                            distill_ii_row.append(eachpred)
                            distill_ii_col.append(iid)
                            distill_ii_row.append(eachpred)
                            distill_ii_col.append(iid)
                            pred_val = ii_pred[batch_i, eachpred]
                            if self.uuii_thres > 0:
                                if pred_val > self.uuii_thres:
                                    distill_ii_value.append(1)
                                    distill_ii_value.append(1)
                                else:
                                    distill_ii_value.append(0)
                                    distill_ii_value.append(0)
                            else:
                                distill_ii_value.append(pred_val)
                                distill_ii_value.append(pred_val)

        return [[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value],
                [distill_ii_row, distill_ii_col, distill_ii_value]]

    def reset_ui_dataset(self, newdata):
        [newuidata, newuudata, newiidata] = newdata

        new_uu_row, new_uu_col, new_uu_val = newuudata
        self.ui_dataset.UserUserNet = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)),
                                                 shape=(self.n_users, self.n_users))
        self.ui_dataset.UserUserNet.setdiag(0)

        new_ii_row, new_ii_col, new_ii_val = newiidata
        self.ui_dataset.ItemItemNet = csr_matrix((new_ii_val, (new_ii_row, new_ii_col)),
                                                 shape=(self.n_items, self.n_items))
        self.ui_dataset.ItemItemNet.setdiag(0)


class WORK2(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.ui_gcn = model.LightGCN()
        # self.matrix_resample = MatrixResample(self.ui_dataset, self.n_users, self.n_items)
        self.matrix_rebuild = MatrixRebuild(self.embedding_dim, self.n_users, self.n_items)

    def prepare_each_epoch(self):
        self.matrix_rebuild.prepare_each_epoch()

    def calculate_embedding(self):
        return self.ui_gcn(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        eu, ei, g0 = self.embedding_user.weight, self.embedding_item.weight, self.Graph
        zu_g0, zi_g0 = self.ui_gcn(eu, ei, g0)

        g1 = self.matrix_rebuild(torch.concat([zu_g0, zi_g0], dim=0), g0)
        zu_g1, zi_g1 = self.ui_gcn(eu, ei, g1)

        g2 = self.matrix_rebuild(torch.concat([zu_g0, zi_g0], dim=0), g0)
        zu_g2, zi_g2 = self.ui_gcn(eu, ei, g2)

        loss = {}
        loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_g0, zi_g0, users, pos, neg)
        loss[losses.Loss.SSL.value] = losses.loss_info_nce(zu_g1, zu_g2, users)
        loss[losses.Loss.SSL.value] += losses.loss_info_nce(zi_g1, zi_g2, pos)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(eu, ei, users, pos, neg)
        return loss
