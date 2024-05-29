import math

import torch
import copy
import numpy as np
import torch.nn as nn
from torch.distributions import RelaxedBernoulli
import scipy.sparse as sp
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax as scatter_softmax

import model
import world
from train import losses, utils, dataloader
from scipy.sparse import csr_matrix
from torch.utils.data.dataloader import default_collate


# class MatrixRebuild(nn.Module):
#     """
#     根据输入的emb，得到一个生成的graph_random，其中包含一定的随机噪声
#     """
#
#     def __init__(self, embedding_dim, n_users, n_items):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.n_users = n_users
#         self.n_items = n_items
#
#         self.mlp_edge_model = nn.Sequential(
#             nn.Linear(self.embedding_dim * 2, self.embedding_dim),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim, 1)
#         ).to(world.device)
#
#         self.graph_random = None  # 每个epoch随机产生的额外graph
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0.0)
#
#     def prepare_each_epoch(self):
#         # 交互图的随机噪声，源代码是100000个
#         number = int(100000)
#         rdmUsrs = torch.randint(self.n_users, [number])  # ancs
#         rdmItms1 = torch.randint_like(rdmUsrs, self.n_items)
#         new_idxs = default_collate([rdmUsrs, rdmItms1])
#         new_vals = torch.tensor([0.05] * number)
#         shape = torch.Size((self.n_items + self.n_users, self.n_items + self.n_users))
#         self.graph_random = torch.sparse_coo_tensor(new_idxs, new_vals, shape).to(world.device)
#
#     def forward(self, all_emb, graph):
#         edge_index = graph._indices()
#         src, dst = edge_index[0], edge_index[1]
#         emb_src = all_emb[src]
#         emb_dst = all_emb[dst]
#
#         edge_emb = torch.cat([emb_src, emb_dst], 1)
#
#         edge_logits = self.mlp_edge_model(edge_emb)  # u、i的emb通过神经网络计算得到edge_logits
#         temperature = 1.0
#         bias = 0.0 + 0.0001  # If bias is 0, we run into problems
#         eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)  # [1-bias, bias]
#         gate_inputs = torch.log(eps) - torch.log(1 - eps)  # 均匀噪声转换为gumble噪声
#         gate_inputs = gate_inputs.cuda()
#         gate_inputs = (gate_inputs + edge_logits) / temperature
#         edge_wight = torch.sigmoid(gate_inputs).squeeze()  # 整理到0和1之间
#         mat = self.__build_prob_neighbourhood(edge_wight, 0.9)
#         graph = torch.sparse_coo_tensor(edge_index, mat, self.graph_random.shape).to(world.device)
#         return (graph + self.graph_random).coalesce()
#
#     @staticmethod
#     def __build_prob_neighbourhood(edge_wight, temperature):
#         attention = torch.clamp(edge_wight, 0.01, 0.99)
#         weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
#                                                      probs=attention).rsample()
#         eps = 0.0
#         mask = (weighted_adjacency_matrix > eps).detach().float()
#         weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
#         return weighted_adjacency_matrix


# class MixHop(nn.Module):
#     def __init__(self, ui_dataset, n_users, n_items, embedding_dim):
#         super().__init__()
#         self.ui_dataset = ui_dataset
#         self.n_users = n_users
#         self.n_items = n_items
#         self.embedding_dim = embedding_dim
#
#         self.layers = 3
#         self.model = model.LightGCN()
#         self.agg_layer_mode = 1
#         self.graph_prepare_mode = 1
#
#         self.Graph_0 = None
#         self.Graph_1 = None
#         self.Graph_2 = None
#
#         self.W = nn.Linear(in_features=embedding_dim * 3, out_features=embedding_dim)
#
#         self.prepare_graph()
#         print("Finish prepare MixHop graphs")
#
#     def forward(self, all_users, all_items):
#         """
#         agg_layer_mode: 用于改变层内聚合模式
#         1: 全连接W  2: 平均池化  3: 求和
#         """
#         if not 1 <= self.agg_layer_mode <= 3:
#             raise NotImplementedError("不存在的层内聚合模式")
#
#         zu, zi = all_users, all_items
#         embs_zu = [zu]
#         embs_zi = [zi]
#         for i in range(self.layers):
#             zu_0, zi_0 = self.model(all_users, all_items, self.Graph_0)
#             zu_1, zi_1 = self.model(all_users, all_items, self.Graph_1)
#             zu_2, zi_2 = self.model(all_users, all_items, self.Graph_2)
#
#             if self.agg_layer_mode == 1:
#                 zu = self.W(torch.concat([zu_0, zu_1, zu_2], dim=-1))
#                 zi = self.W(torch.concat([zi_0, zi_1, zi_2], dim=-1))
#             elif self.agg_layer_mode == 2:
#                 zu = torch.mean(torch.stack([zu_0, zu_1, zu_2], dim=0), dim=0)
#                 zi = torch.mean(torch.stack([zi_0, zi_1, zi_2], dim=0), dim=0)
#             elif self.agg_layer_mode == 3:
#                 zu = torch.sum(torch.stack([zu_0, zu_1, zu_2], dim=0), dim=0)
#                 zi = torch.sum(torch.stack([zi_0, zi_1, zi_2], dim=0), dim=0)
#
#             embs_zu.append(zu)
#             embs_zi.append(zi)
#
#         return torch.mean(torch.stack(embs_zu, dim=0), dim=0), torch.mean(torch.stack(embs_zi, dim=0), dim=0)
#
#     def prepare_graph(self):
#         """
#         graph_prepare_mode: 用于改变二阶邻接矩阵的生成模式
#         1: 邻接矩阵二次方后值归为1后再做拉普拉斯变换
#         2: 邻接矩阵二次方后做拉普拉斯变换
#         """
#         if not 1 <= self.graph_prepare_mode <= 2:
#             raise NotImplementedError("不存在的二阶邻接矩阵的生成模式")
#
#         n = self.n_items + self.n_users
#         indices = torch.arange(n).unsqueeze(0).repeat(2, 1)
#         values = torch.ones(n)
#         self.Graph_0 = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to(world.device)
#         self.Graph_1 = self.ui_dataset.getSparseGraph()
#         new_values = torch.ones_like(self.Graph_1.values())
#         mat = torch.sparse_coo_tensor(self.Graph_1.indices(), new_values, self.Graph_1.shape)
#         mat = torch.sparse.mm(mat, mat)
#         if self.graph_prepare_mode == 1:
#             mat.values().fill_(1)
#         elif self.graph_prepare_mode == 2:
#             pass
#         try:
#             degree_values = torch.sparse.sum(mat, dim=1).to_dense()
#             degree_inv_sqrt = torch.diag(degree_values.pow(-0.5))  # 内存超了
#             degree_inv_sqrt[degree_inv_sqrt == torch.inf] = 0.
#             degree_inv_sqrt_sparse = degree_inv_sqrt.to_sparse()
#             D_inv_sqrt_A = torch.sparse.mm(degree_inv_sqrt_sparse, mat)
#             self.Graph_2 = torch.sparse.mm(D_inv_sqrt_A, degree_inv_sqrt_sparse)
#         except Exception:
#             indices_np = mat.coalesce().cpu().indices().numpy()
#             values_np = mat.coalesce().cpu().values().numpy()
#             adj_mat = sp.coo_matrix((values_np, (indices_np[0], indices_np[1])), shape=(n, n))
#             adj_mat = adj_mat.todok()
#
#             rowsum = np.array(adj_mat.sum(axis=1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat = sp.diags(d_inv)
#
#             norm_adj = d_mat.dot(adj_mat)
#             norm_adj = norm_adj.dot(d_mat)
#             norm_adj = norm_adj.tocsr()
#
#             self.Graph_2 = utils.convert_sp_mat_to_sp_tensor(norm_adj)
#             self.Graph_2 = self.Graph_2.coalesce().to(world.device)


class MatrixResample:
    """
    根据输入的预训练emb，采样得到一个包含uuii的graph，不包含噪声
    """

    def __init__(self, ui_dataset, n_users, n_items):
        super().__init__()
        self.ui_dataset = copy.deepcopy(ui_dataset)
        self.n_users = n_users
        self.n_items = n_items

        use_pretrain = True
        self.sample_batch_size = 100
        self.distill_userK = 1  # <= 50
        self.distill_itemK = 1  # <= 50
        self.distill_uuK = 4
        self.distill_iiK = 1
        self.distill_thres = 0.8
        self.uuii_thres = 0.8
        self.mode = world.hyper_WORK2_resample_mode
        self.new_ui_mode = world.hyper_WORK2_resample_new_ui_mode

        self.f = nn.Sigmoid()
        if use_pretrain:
            emb = torch.load(world.PATH_PRETRAIN + '/' + world.dataset + '_' + world.pretrain_input + '.pretrain')
            self.eu_ui = torch.nn.Embedding.from_pretrained(emb['embedding_user.weight']).weight
            self.ei_ui = torch.nn.Embedding.from_pretrained(emb['embedding_item.weight']).weight
        else:
            self.eu_ui = None
            self.ei_ui = None

    def prepare_each_epoch(self, eu=None, ei=None, eu_ckg=None, ei_ckg=None):
        if eu is not None:
            self.eu_ui = eu
        if ei is not None:
            self.ei_ui = ei
        sampled_ui = self.__sample_ui_topk(eu_ckg, ei_ckg)
        self.__reset_ui_dataset(sampled_ui)
        graph = self.ui_dataset.getSparseGraph(include_uuii=True, regenerate_not_save=True)
        return graph

    def __sample_ui_topk(self, eu_ckg=None, ei_ckg=None):
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
                    rating_pred = self.__get_batch_ratings(self.eu_ui, batch_users_gpu, self.ei_ui)
                    uu_pred = self.__get_batch_ratings(self.eu_ui, batch_users_gpu, self.eu_ui)

                    if eu_ckg is not None and ei_ckg is not None:
                        rating_pred_ckg = self.__get_batch_ratings(eu_ckg, batch_users_gpu, ei_ckg)
                        uu_pred_ckg = self.__get_batch_ratings(eu_ckg, batch_users_gpu, eu_ckg)
                        if self.mode == 1:
                            rating_pred = rating_pred_ckg - rating_pred
                            uu_pred = uu_pred_ckg - uu_pred
                        else:
                            rating_pred = rating_pred_ckg
                            uu_pred = uu_pred_ckg

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
                                    distill_value.append(pred_val)
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
                                    distill_uu_value.append(pred_val)
                                    distill_uu_value.append(pred_val)
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

                    rating_pred = self.__get_batch_ratings(self.ei_ui, batch_items_gpu,
                                                           self.eu_ui)
                    ii_pred = self.__get_batch_ratings(self.ei_ui, batch_items_gpu,
                                                       self.ei_ui)

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
                                    distill_value.append(pred_val)
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
                                    distill_ii_value.append(pred_val)
                                    distill_ii_value.append(pred_val)
                                else:
                                    distill_ii_value.append(0)
                                    distill_ii_value.append(0)
                            else:
                                distill_ii_value.append(pred_val)
                                distill_ii_value.append(pred_val)

        return [[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value],
                [distill_ii_row, distill_ii_col, distill_ii_value]]

    def __get_batch_ratings(self, all_batch_emb, batch, all_emb):
        batch_emb = all_batch_emb[batch.long()]
        ratings = self.f(torch.matmul(batch_emb, all_emb.t()))
        return ratings

    def __reset_ui_dataset(self, newdata):
        [newuidata, newuudata, newiidata] = newdata
        # new_row, new_col, new_val = newuidata
        # if self.new_ui_mode == 2:
        #     self.ui_dataset.UserItemNet = csr_matrix((new_val, (new_row, new_col)),
        #                                              shape=(self.n_users, self.n_items))
        # elif self.new_ui_mode == 3:
        #     new_ui = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_users, self.n_items))
        #     self.ui_dataset.UserItemNet = self.ui_dataset.UserItemNet + new_ui

        new_uu_row, new_uu_col, new_uu_val = newuudata
        add_uu_net = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)), shape=(self.n_users, self.n_users))
        self.ui_dataset.UserUserNet = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)),
                                                 shape=(self.n_users, self.n_users))
        self.ui_dataset.UserUserNet.setdiag(0)

        # new_ii_row, new_ii_col, new_ii_val = newiidata
        # self.ui_dataset.ItemItemNet = csr_matrix((new_ii_val, (new_ii_row, new_ii_col)),
        #                                          shape=(self.n_items, self.n_items))
        # self.ui_dataset.ItemItemNet.setdiag(0)


class CKGGCN(nn.Module):
    def __init__(self, dims, n_relations):
        super(CKGGCN, self).__init__()
        self.relation_emb = nn.Parameter(nn.init.normal_(torch.empty(n_relations, dims), std=0.1))

        self.W_Q = nn.Parameter(nn.init.normal_(torch.Tensor(dims, dims), std=0.1))

        self.n_heads = 2
        self.d_k = dims // self.n_heads

    def _agg_layer(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        # attention from entity to item/entity
        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        relation_emb = self.relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
        entity_agg = F.normalize(entity_agg)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)

        return entity_agg, user_agg

    def forward(self, layers_num, user_emb, entity_emb, inter_edge, inter_edge_w, edge_index, edge_type):
        user_embs = [user_emb]
        entity_embs = [entity_emb]
        for i in range(layers_num):
            entity_emb, user_emb = self._agg_layer(user_emb, entity_emb,
                                                   edge_index, edge_type,
                                                   inter_edge, inter_edge_w)
            user_embs.append(user_emb)
            entity_embs.append(entity_emb)
        user_embs = torch.mean(torch.stack(user_embs, dim=1), dim=1)
        entity_embs = torch.mean(torch.stack(entity_embs, dim=1), dim=1)
        return user_embs, entity_embs


class WORK2(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.kg_dataset = dataloader.KGDataset()
        self.n_relations = self.kg_dataset.relation_count * 2 - 1
        self.num_entities = self.kg_dataset.entity_count - 1  # exclude items
        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities, embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_entity.weight, std=0.1)

        self.inter_edge_w = self.Graph.values()[:self.Graph.values().shape[0] // 2]
        self.inter_edge = [self.Graph.indices()[0, :self.Graph.indices()[0].shape[0] // 2],
                           self.Graph.indices()[1, :self.Graph.indices()[0].shape[0] // 2] - self.n_users]
        self.inter_edge = torch.stack(self.inter_edge, dim=0)

        self.Graph_KG = self.kg_dataset.get_kg_graph(1, True)
        self.edge_type = torch.LongTensor(self.Graph_KG.data).to(world.device)
        self.edge_index = torch.LongTensor(np.stack((self.Graph_KG.row, self.Graph_KG.col))).to(world.device)

        self.ui_gcn = model.LightGCN()
        self.ckg_gcn = CKGGCN(dims=self.embedding_dim, n_relations=self.n_relations)

        self.ui_layers = 3
        self.ckg_layers = 3

        self.matrix_resample = MatrixResample(self.ui_dataset, self.n_users, self.n_items)
        self.Graph_resample = None  # Prepare each epoch

        self.Graph_resample = self.matrix_resample.prepare_each_epoch()
    # def prepare_each_epoch(self):
    #     if world.hyper_WORK2_reset_ui_graph:
    #         eu, ei, ee, g0 = (self.embedding_user.weight,
    #                           self.embedding_item.weight,
    #                           self.embedding_entity.weight,
    #                           self.Graph)
    #         zu_ui, zi_ui = self.ui_gcn(eu, ei, g0)
    #         zu_ckg, zi_ckg = self.ckg_gcn(self.ckg_layers,
    #                                       eu,
    #                                       torch.concat([ei, ee]),
    #                                       self.inter_edge,
    #                                       self.inter_edge_w,
    #                                       self.edge_index,
    #                                       self.edge_type)
    #         self.Graph_resample = self.matrix_resample.prepare_each_epoch(zu_ui, zi_ui, zu_ckg, zi_ckg[:self.n_items])
    #     else:
    #         self.Graph_resample = self.Graph
    #
    #     if world.hyper_WORK2_reset_ckg_graph:
    #         self.inter_edge_w = self.Graph_resample.values()[:self.Graph.values().shape[0] // 2]
    #         self.inter_edge = [self.Graph_resample.indices()[0, :self.Graph.indices()[0].shape[0] // 2],
    #                            self.Graph_resample.indices()[1, :self.Graph.indices()[0].shape[0] // 2] - self.n_users]
    #         self.inter_edge = torch.stack(self.inter_edge, dim=0)

    def calculate_embedding(self):
        eu, ei = self.embedding_user.weight, self.embedding_item.weight
        zu_g0, zi_g0 = self.ui_gcn(eu, ei, self.Graph_resample)
        zu_g1, zi_g1 = self.ckg_gcn(self.ckg_layers,
                                    self.embedding_user.weight,
                                    torch.concat([self.embedding_item.weight, self.embedding_entity.weight]),
                                    self.inter_edge,
                                    self.inter_edge_w,
                                    self.edge_index,
                                    self.edge_type)
        if self.training:
            return zu_g0 + zu_g1, zi_g0 + zi_g1[:self.n_items], zu_g0, zu_g1, zi_g0, zi_g1[:self.n_items]
        else:
            if world.hyper_WORK2_BPR_mode == 1:
                return zu_g0, zi_g0
            elif world.hyper_WORK2_BPR_mode == 2:
                return zu_g1, zi_g1
            else:
                return zu_g0 + zu_g1, zi_g0 + zi_g1[:self.n_items]

    def calculate_loss(self, users, pos, neg):
        eu, ei, g0 = self.embedding_user.weight, self.embedding_item.weight, self.Graph
        zu, zi, zu_g0, zu_g1, zi_g0, zi_g1 = self.calculate_embedding()

        loss = dict()
        if world.hyper_WORK2_BPR_mode == 1:
            loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_g0, zi_g0, users, pos, neg)
        elif world.hyper_WORK2_BPR_mode == 2:
            loss[losses.Loss.BPR.value] = losses.loss_BPR(zu_g1, zi_g1, users, pos, neg)
        else:
            loss[losses.Loss.BPR.value] = losses.loss_BPR(zu, zi, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(eu, ei, users, pos, neg)

        if not world.hyper_WORK2_SSM_use:
            return loss

        if world.hyper_WORK2_SSM_mode == 1:
            loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g0, zi_g0, users, pos)
        elif world.hyper_WORK2_SSM_mode == 2:
            loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g1, zi_g1, users, pos)
        else:
            loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu, zi, users, pos)
        return loss
