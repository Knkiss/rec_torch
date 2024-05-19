import torch
import numpy as np

import model
import world
from train import losses, utils
from scipy.sparse import csr_matrix


class GraphDA(model.AbstractRecModel):
    def __init__(self):
        super().__init__()

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

        self.reset_ui_dataset(self.sample_ui_from_pretrained())
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
                    raise ValueError(f"sample_batch_size is too big for this dataset, try a small one {len(users) // 10}")

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
        new_row, new_col, new_val = newuidata
        self.ui_dataset.UserItemNet = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_users, self.n_items))
        self.ui_dataset.users_D = np.array(self.ui_dataset.UserItemNet.sum(axis=1)).squeeze()
        self.ui_dataset.users_D[self.ui_dataset.users_D == 0.] = 1.
        self.ui_dataset.items_D = np.array(self.ui_dataset.UserItemNet.sum(axis=0)).squeeze()
        self.ui_dataset.items_D[self.ui_dataset.items_D == 0.] = 1.

        new_uu_row, new_uu_col, new_uu_val = newuudata
        self.ui_dataset.UserUserNet = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)), shape=(self.n_users, self.n_users))
        self.ui_dataset.UserUserNet.setdiag(0)

        new_ii_row, new_ii_col, new_ii_val = newiidata
        self.ui_dataset.ItemItemNet = csr_matrix((new_ii_val, (new_ii_row, new_ii_col)), shape=(self.n_items, self.n_items))
        self.ui_dataset.ItemItemNet.setdiag(0)

    def calculate_embedding(self):
        return self.forward(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                    pos, neg)
        return loss

    def forward(self, all_users, all_items, graph, dropout=True, drop_prob=0.2, n_layers=3, output_one_layer=False):
        num_users = all_users.shape[0]
        num_items = all_items.shape[0]
        all_emb = torch.cat([all_users, all_items])
        embs = [all_emb]
        if dropout and self.training:
            g_dropped = utils.dropout_x(graph, 1-drop_prob)
        else:
            g_dropped = graph
        for layer in range(n_layers):
            all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)

        if not output_one_layer:
            embs = torch.stack(embs, dim=1)
            all_emb = torch.mean(embs, dim=1)
            return torch.split(all_emb, [num_users, num_items])
        else:
            return torch.split(embs[-1], [num_users, num_items])
