import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax

import model
import world
from train import losses, dataloader


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


# class MatrixResample:
#     """
#     根据输入的预训练emb，采样得到一个包含uuii的graph，不包含噪声
#     """
#
#     def __init__(self, ui_dataset, kg_graph, n_users, n_items, n_entities):
#         super().__init__()
#         self.ui_dataset = copy.deepcopy(ui_dataset)
#         self.n_users = n_users
#         self.n_items = n_items
#
#         use_pretrain = False  # set to parameters
#         self.sample_batch_size = 100
#         self.distill_userK = world.WORK2_sample_uiK  # <= 50
#         self.distill_itemK = world.WORK2_sample_iuK  # <= 50
#         self.distill_uuK = world.WORK2_sample_uuK  # 5 with 0.8 get best
#         self.distill_iiK = world.WORK2_sample_iiK
#         self.distill_thres = 0.8
#         self.uu_gate = 0.8
#         self.ii_gate = 0.8
#         self.mode = 1  # 重采样矩阵的值根据什么预测结果排序 1=ckg-ui  2=ckg
#         self.new_ui_mode = 3  # 重采样矩阵的新得到的UI如何使用 1=dont use 2=replace 3=add
#
#         self.f = nn.Sigmoid()
#         if use_pretrain and world.hyper_WORK2_reset_ui_graph:
#             emb = torch.load(world.PATH_PRETRAIN + '/' + world.dataset + '_' + world.pretrain_input + '.pretrain')
#             self.eu_ui = torch.nn.Embedding.from_pretrained(emb['embedding_user.weight']).weight
#             self.ei_ui = torch.nn.Embedding.from_pretrained(emb['embedding_item.weight']).weight
#         else:
#             self.eu_ui = None
#             self.ei_ui = None
#
#         self.ii_matrix = None
#         self.uu_matrix = None
#         self.prepare_init(kg_graph, n_items, n_entities)
#
#     def prepare_init(self, kg_graph, n_items, n_entities):
#         # 根据kg图或者ui图，构造确定性的二阶ii关系
#
#         construct_mode = 3  # world设置超参数
#         if construct_mode == 1:
#             print("从所有ie关系中构建ii二阶关系")  # 得到的ii二阶关系非常少
#             row = kg_graph.row
#             col = kg_graph.col
#             data = kg_graph.data
#             mask = (row < n_items) & (col >= n_items)
#             ie_matrix = coo_matrix((np.ones_like(data[mask]), (row[mask], col[mask] - n_items)),
#                                   (n_items, n_entities)).tocsr()
#             self.ii_matrix = ie_matrix.dot(ie_matrix.transpose()).tocoo()
#             self.ii_matrix.data = np.ones_like(self.ii_matrix.data, dtype=np.float32)  # 5195595个二阶ii关系，原矩阵为5508409
#         elif construct_mode == 2:
#             print("从单个ie关系中构建ii二阶关系后融合")
#             row = kg_graph.row
#             col = kg_graph.col
#             data = kg_graph.data
#             start_r, end_r = data.min(), data.max()
#             ii_matrices = []
#             for r in range(start_r, end_r+1):
#                 mask = (row < n_items) & (col >= n_items) & (data == r)
#                 ie_matrix = coo_matrix((np.ones_like(data[mask]), (row[mask], col[mask] - n_items)),
#                                       (n_items, n_entities)).tocsr()
#                 ii_matrix = ie_matrix.dot(ie_matrix.transpose())
#                 ii_matrix.data = np.ones_like(ii_matrix.data, dtype=np.float32)
#                 ii_matrices.append(ii_matrix)
#             self.ii_matrix = sum(ii_matrices)  # 计算结果与所有relation计算的一致
#         elif construct_mode == 3:
#             print("从ui关系中计算二阶ii关系和二阶uu关系")
#             self.uu_matrix = self.ui_dataset.UserItemNet.dot(self.ui_dataset.UserItemNet.transpose()).tocoo()
#             self.ii_matrix = self.ui_dataset.UserItemNet.transpose().dot(self.ui_dataset.UserItemNet).tocoo()
#         else:
#             raise NotImplementedError("错误的construct_mode设置")
#
#         rowsum = np.array(self.ii_matrix.sum(1))
#         d_inv = np.power(rowsum, -0.5).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#
#         # d_mat = sp.diags(d_inv)
#         # norm_adj = d_mat.dot(self.ii_matrix)
#         # norm_adj = norm_adj.dot(d_mat)
#         # ii_norm_adj = norm_adj.tocsr()
#         # self.ii_graph = utils.convert_sp_mat_to_sp_tensor(ii_norm_adj).coalesce().to(world.device)
#
#         d_mat_inv = sp.diags(d_inv)
#         ii_mean_adj = d_mat_inv.dot(self.ii_matrix)
#         self.ii_graph = utils.convert_sp_mat_to_sp_tensor(ii_mean_adj).coalesce().to(world.device)
#
#     def prepare_init_from_pretrain(self):
#         sampled_ui = self.__sample_ui_topk()
#         self.__reset_ui_dataset(sampled_ui)
#         graph = self.ui_dataset.getSparseGraph(include_uuii=True, regenerate_not_save=True, regenerate_d=False)
#         return graph
#
#     # def prepare_each_epoch(self, eu_ckg, ei_ckg):
#     #     if self.uu_matrix is not None:
#     #         pass
#     #
#     #     ei_ckg_gcn = (torch.matmul(self.ii_graph, ei_ckg) + ei_ckg) / 2
#     #     a = ei_ckg.cpu().detach().numpy()
#     #     b = ei_ckg_gcn.cpu().detach().numpy()
#     #
#     #     ei_ckg_score = F.cosine_similarity(ei_ckg.unsqueeze(dim=0), ei_ckg.unsqueeze(dim=1), dim=2)
#     #     ei_ckg_gcn_score = F.cosine_similarity(ei_ckg_gcn.unsqueeze(dim=0), ei_ckg_gcn.unsqueeze(dim=1), dim=2)
#     #     a_score = ei_ckg_score.cpu().detach().numpy()
#     #     b_score = ei_ckg_gcn_score.cpu().detach().numpy()
#     #     print(1)
#
#
#     def __sample_ui_topk(self, eu_ckg=None, ei_ckg=None):
#         distill_user_row = []
#         distill_item_col = []
#         distill_value = []
#
#         distill_uu_row = []
#         distill_uu_col = []
#         distill_uu_value = []
#
#         distill_ii_row = []
#         distill_ii_col = []
#         distill_ii_value = []
#
#         u_batch_size = self.sample_batch_size
#
#         if self.distill_userK > 0 or self.distill_uuK > 0:
#             with torch.no_grad():
#                 users = list(set(self.ui_dataset.trainUser))
#                 try:
#                     assert u_batch_size <= len(users) / 10
#                 except AssertionError:
#                     raise ValueError(
#                         f"sample_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#
#                 for batch_users in utils.minibatch(users, batch_size=u_batch_size):
#                     batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
#                     rating_pred = self.__get_batch_ratings(self.eu_ui, batch_users_gpu, self.ei_ui)
#                     uu_pred = self.__get_batch_ratings(self.eu_ui, batch_users_gpu, self.eu_ui)
#
#                     if eu_ckg is not None and ei_ckg is not None:
#                         rating_pred_ckg = self.__get_batch_ratings(eu_ckg, batch_users_gpu, ei_ckg)
#                         uu_pred_ckg = self.__get_batch_ratings(eu_ckg, batch_users_gpu, eu_ckg)
#                         if self.mode == 1:
#                             rating_pred = rating_pred_ckg - rating_pred
#                             uu_pred = uu_pred_ckg - uu_pred
#                         else:
#                             rating_pred = rating_pred_ckg
#                             uu_pred = uu_pred_ckg
#
#                     rating_pred = rating_pred.cpu().data.numpy().copy()
#                     ind = np.argpartition(rating_pred, -50)[:, -50:]
#                     arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
#                     arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
#                     batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
#
#                     uu_pred = uu_pred.cpu().data.numpy().copy()
#                     uu_ind = np.argpartition(uu_pred, -50)[:, -50:]
#                     uu_arr_ind = uu_pred[np.arange(len(uu_pred))[:, None], uu_ind]
#                     uu_arr_ind_argsort = np.argsort(uu_arr_ind)[np.arange(len(uu_pred)), ::-1]
#                     uu_batch_pred_list = uu_ind[np.arange(len(uu_pred))[:, None], uu_arr_ind_argsort]
#
#                     partial_batch_pred_list = batch_pred_list[:, :self.distill_userK]
#                     uu_partial_batch_pred_list = uu_batch_pred_list[:, :self.distill_uuK]
#
#                     for batch_i in range(partial_batch_pred_list.shape[0]):
#                         uid = batch_users[batch_i]
#                         user_pred = partial_batch_pred_list[batch_i]
#                         uu_user_pred = uu_partial_batch_pred_list[batch_i]
#                         for eachpred in user_pred:
#                             distill_user_row.append(uid)
#                             distill_item_col.append(eachpred)
#                             pred_val = rating_pred[batch_i, eachpred]
#                             if self.distill_thres > 0:
#                                 if pred_val > self.distill_thres:
#                                     distill_value.append(pred_val)
#                                 else:
#                                     distill_value.append(0)
#                             else:
#                                 distill_value.append(pred_val)
#
#                         for eachpred in uu_user_pred:
#                             distill_uu_row.append(uid)
#                             distill_uu_col.append(eachpred)
#                             distill_uu_row.append(eachpred)
#                             distill_uu_col.append(uid)
#                             pred_val = uu_pred[batch_i, eachpred]
#                             if self.uu_gate > 0:
#                                 if pred_val > self.uu_gate:
#                                     distill_uu_value.append(pred_val)
#                                     distill_uu_value.append(pred_val)
#                                 else:
#                                     distill_uu_value.append(0)
#                                     distill_uu_value.append(0)
#                             else:
#                                 distill_uu_value.append(pred_val)
#                                 distill_uu_value.append(pred_val)
#
#         if self.distill_itemK > 0 or self.distill_iiK > 0:
#             with torch.no_grad():
#                 items = [i for i in range(self.n_items)]
#                 total_batch = len(items) // u_batch_size + 1
#                 for batch_items in utils.minibatch(items, batch_size=u_batch_size):
#                     batch_items_gpu = torch.Tensor(batch_items).long().to(world.device)
#
#                     rating_pred = self.__get_batch_ratings(self.ei_ui, batch_items_gpu,
#                                                            self.eu_ui)
#                     ii_pred = self.__get_batch_ratings(self.ei_ui, batch_items_gpu,
#                                                        self.ei_ui)
#
#                     rating_pred = rating_pred.cpu().data.numpy().copy()
#                     ind = np.argpartition(rating_pred, -50)[:, -50:]
#                     arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
#                     arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
#                     batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
#
#                     ii_pred = ii_pred.cpu().data.numpy().copy()
#                     ii_ind = np.argpartition(ii_pred, -50)[:, -50:]
#                     ii_arr_ind = ii_pred[np.arange(len(ii_pred))[:, None], ii_ind]
#                     ii_arr_ind_argsort = np.argsort(ii_arr_ind)[np.arange(len(ii_pred)), ::-1]
#                     ii_batch_pred_list = ii_ind[np.arange(len(ii_pred))[:, None], ii_arr_ind_argsort]
#
#                     partial_batch_pred_list = batch_pred_list[:, :self.distill_itemK]
#                     ii_partial_batch_pred_list = ii_batch_pred_list[:, :self.distill_iiK]
#                     for batch_i in range(partial_batch_pred_list.shape[0]):
#                         iid = batch_items[batch_i]
#                         item_pred = partial_batch_pred_list[batch_i]
#                         ii_item_pred = ii_partial_batch_pred_list[batch_i]
#                         for eachpred in item_pred:
#                             distill_user_row.append(eachpred)
#                             distill_item_col.append(iid)
#                             pred_val = rating_pred[batch_i, eachpred]
#                             if self.distill_thres > 0:
#                                 if pred_val > self.distill_thres:
#                                     distill_value.append(pred_val)
#                                 else:
#                                     distill_value.append(0)
#                             else:
#                                 distill_value.append(pred_val)
#
#                         for eachpred in ii_item_pred:
#                             distill_ii_row.append(eachpred)
#                             distill_ii_col.append(iid)
#                             distill_ii_row.append(eachpred)
#                             distill_ii_col.append(iid)
#                             pred_val = ii_pred[batch_i, eachpred]
#                             if self.ii_gate > 0:
#                                 if pred_val > self.ii_gate:
#                                     distill_ii_value.append(pred_val)
#                                     distill_ii_value.append(pred_val)
#                                 else:
#                                     distill_ii_value.append(0)
#                                     distill_ii_value.append(0)
#                             else:
#                                 distill_ii_value.append(pred_val)
#                                 distill_ii_value.append(pred_val)
#
#        return [[distill_user_row, distill_item_col, distill_value],
#                [distill_uu_row, distill_uu_col, distill_uu_value],
#                [distill_ii_row, distill_ii_col, distill_ii_value]]
#
#     def __get_batch_ratings(self, all_batch_emb, batch, all_emb):
#         cosine_or_sigmoid = 1
#         if cosine_or_sigmoid == 1:
#             all_batch_emb = F.normalize(all_batch_emb, dim=1)
#             all_emb = F.normalize(all_emb, dim=1)
#             batch_emb = all_batch_emb[batch.long()]
#             ratings = torch.matmul(batch_emb, all_emb.t())
#         else:
#             batch_emb = all_batch_emb[batch.long()]
#             ratings = self.f(torch.matmul(batch_emb, all_emb.t()))
#         return ratings
#
#     def __reset_ui_dataset(self, newdata):
#         [newuidata, newuudata, newiidata] = newdata
#         new_row, new_col, new_val = newuidata
#         add_ui_net = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_users, self.n_items))
#         add_ui_net.eliminate_zeros()
#         if self.new_ui_mode == 2:
#             self.ui_dataset.UserItemNet = add_ui_net
#         elif self.new_ui_mode == 3:
#             self.ui_dataset.UserItemNet = self.ui_dataset.UserItemNet + add_ui_net
#             self.ui_dataset.UserItemNet[self.ui_dataset.UserItemNet > 1] = 1
#
#         # UU关系补充，取distill_uuK * 2（包含反向关系）的uu关系补充进来，值的范围在[self.uu_gate, 1]
#         new_uu_row, new_uu_col, new_uu_val = newuudata
#         add_uu_net = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)), shape=(self.n_users, self.n_users))
#         add_uu_net.setdiag(0)
#         add_uu_net[add_uu_net > 1] = 1  # 存疑？重复选择到相同关系时，如何处理？
#         add_uu_net.eliminate_zeros()
#         self.ui_dataset.UserUserNet = add_uu_net
#
#         # UU关系补充，取distill_iiK * 2（包含反向关系）的ii关系补充进来，值的范围在[self.ii_gate, 1]
#         new_ii_row, new_ii_col, new_ii_val = newiidata
#         add_ii_net = csr_matrix((new_ii_val, (new_ii_row, new_ii_col)), shape=(self.n_items, self.n_items))
#         add_ii_net.setdiag(0)
#         add_ii_net[add_ii_net > 1] = 1
#         add_ii_net.eliminate_zeros()
#         self.ui_dataset.ItemItemNet = add_ii_net


class CKGGCN(nn.Module):
    def __init__(self, dims, n_relations):
        super(CKGGCN, self).__init__()
        self.relation_emb = nn.Parameter(nn.init.normal_(torch.empty(n_relations, dims), std=0.1))

        self.W_Q = nn.Parameter(nn.init.normal_(torch.Tensor(dims, dims), std=0.1))

        self.n_heads = 2
        self.d_k = dims // self.n_heads

    def _agg_layer(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        head, tail = edge_index
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]

        # attention from entity to item/entity
        query = (head_emb @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (tail_emb @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        relation_emb = self.relation_emb[edge_type - 1]
        neigh_relation_emb = tail_emb * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        entity_agg_res = torch.zeros_like(entity_emb)
        entity_agg = entity_agg_res.index_add_(0, head, entity_agg)
        entity_agg = F.normalize(entity_agg)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = torch.zeros_like(user_emb)
        user_agg = user_agg.index_add_(0, inter_edge[0, :], item_agg)
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

        # self.matrix_resample = MatrixResample(self.ui_dataset,
        #                                       self.Graph_KG,
        #                                       self.n_users,
        #                                       self.n_items,
        #                                       self.num_entities)
        self.Graph_resample = self.Graph  # Prepare each epoch

        # Use for KD on item group
        self.group_mlp = nn.Linear(in_features=self.embedding_dim,
                                   out_features=world.hyper_WORK2_cluster_num).to(world.device)
        nn.init.normal_(self.group_mlp.weight, std=0.1)

        # if world.hyper_WORK2_reset_ui_graph:
        #     self.Graph_resample = self.matrix_resample.prepare_init_from_pretrain()

    # def prepare_each_epoch(self):
    #     eu, ei, ee, g0 = (self.embedding_user.weight,
    #                       self.embedding_item.weight,
    #                       self.embedding_entity.weight,
    #                       self.Graph)
    #     zu_ui, zi_ui = self.ui_gcn(eu, ei, g0)
    #     zu_ckg, zi_ckg = self.ckg_gcn(self.ckg_layers,
    #                                   eu,
    #                                   torch.concat([ei, ee]),
    #                                   self.inter_edge,
    #                                   self.inter_edge_w,
    #                                   self.edge_index,
    #                                   self.edge_type)
    #
    #     self.matrix_resample.prepare_each_epoch(zu_ckg, zi_ckg[:self.n_items])

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
        zi_g1 = zi_g1[:self.n_items]

        if self.training:
            return zu_g0 + zu_g1, zi_g0 + zi_g1, zu_g0, zu_g1, zi_g0, zi_g1
        else:
            if world.hyper_WORK2_BPR_mode == 1:
                return zu_g0, zi_g0
            elif world.hyper_WORK2_BPR_mode == 2:
                return zu_g1, zi_g1
            else:
                return zu_g0 + zu_g1, zi_g0 + zi_g1

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

        if world.hyper_WORK2_SSM_mode > 0:
            if world.hyper_WORK2_SSM_mode == 1:
                loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g0, zi_g0, users, pos)
            elif world.hyper_WORK2_SSM_mode == 2:
                loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu_g1, zi_g1, users, pos)
            elif world.hyper_WORK2_SSM_mode == 3:
                loss[losses.Loss.SSL.value] = losses.loss_SSM_origin(zu, zi, users, pos)
            else:
                raise NotImplementedError("world.hyper_WORK2_SSM_mode")

        if world.hyper_WORK2_KD_mode > 0:
            if world.hyper_WORK2_KD_mode == 1:
                loss[losses.Loss.MAE.value] = losses.loss_kd_ii_graph_batch(zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 2:
                loss[losses.Loss.MAE.value] = losses.loss_kd_ii_graph_batch(zi_g1, zi_g0, pos, neg)  # w/ neg
            elif world.hyper_WORK2_KD_mode == 3:
                loss[losses.Loss.MAE.value] = losses.loss_kd_cluster_ii_graph_batch(zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 4:
                loss[losses.Loss.MAE.value] = losses.loss_kd_cluster_ii_graph_batch(zi_g1, zi_g0)
            elif world.hyper_WORK2_KD_mode == 5:
                loss[losses.Loss.MAE.value] = losses.loss_kd_A_graph_batch(zu_g1, zu_g0, zi_g1, zi_g0, users, pos)
            elif world.hyper_WORK2_KD_mode == 6:
                loss[losses.Loss.MAE.value] = losses.loss_kd_mlp_ii_graph_batch(self.group_mlp, zi_g1, zi_g0, pos)
            elif world.hyper_WORK2_KD_mode == 7:
                loss[losses.Loss.MAE.value] = losses.loss_kd_mlp_ii_graph_batch(self.group_mlp, zi_g1, zi_g0)
            elif world.hyper_WORK2_KD_mode == 8:
                loss[losses.Loss.MAE.value] = losses.loss_bpr_mlp_ui_graph_batch(self.group_mlp, zi_g1, zi_g0,
                                                                                 zu_g1, zu_g0, users, pos, neg,
                                                                                 form='BPR')
            elif world.hyper_WORK2_KD_mode == 9:
                loss[losses.Loss.MAE.value] = losses.loss_bpr_mlp_ui_graph_batch(self.group_mlp, zi_g1, zi_g0,
                                                                                 zu_g1, zu_g0, users, pos, neg,
                                                                                 form='InfoNCE')
            else:
                raise NotImplementedError("world.hyper_WORK2_KD_mode")

        return loss
