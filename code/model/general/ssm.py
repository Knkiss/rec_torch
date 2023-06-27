import model
import world
from train import losses


class SSM(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.model = model.LightGCN()

    def calculate_embedding(self):
        return self.model(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_SSM_origin(all_users, all_items, users, pos)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos,
                                                             neg)
        return loss
