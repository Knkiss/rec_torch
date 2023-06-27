import model
from train import losses
from train.losses import Loss


class MF(model.AbstractRecModel):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos,
                                                             neg)
        return loss

    def calculate_embedding(self):
        return self.embedding_user.weight, self.embedding_item.weight