import model
from train import losses, utils


class SGL(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.graph_1 = None
        self.graph_2 = None
        self.model = model.LightGCN()

    def prepare_each_epoch(self):
        self.graph_1 = utils.create_adj_mat(self.ui_dataset.trainUser, self.ui_dataset.trainItem,
                                            self.num_users, self.num_items, is_subgraph=True)
        self.graph_2 = utils.create_adj_mat(self.ui_dataset.trainUser, self.ui_dataset.trainItem,
                                            self.num_users, self.num_items, is_subgraph=True)

    def calculate_embedding(self):
        return self.model(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                    pos, neg)
        users_1, items_1, users_2, items_2 = self.forward(self.embedding_user.weight, self.embedding_item.weight,
                                                          self.graph_1, self.graph_2)
        loss_ssl_item = losses.loss_info_nce(items_1, items_2, pos)
        loss_ssl_user = losses.loss_info_nce(users_1, users_2, users)
        loss[losses.Loss.SSL.value] = loss_ssl_user + loss_ssl_item
        return loss

    def forward(self, all_users, all_items, graph_1, graph_2):
        users_1, items_1 = self.model(all_users, all_items, graph_1)
        users_2, items_2 = self.model(all_users, all_items, graph_2)
        return users_1, items_1, users_2, items_2
