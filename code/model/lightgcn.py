import torch

import model
from train import losses, utils

N_LAYERS = 3
KEEP_PROB = 0.8
DROPOUT = True

class LightGCN(model.AbstractRecModel):
    def __init__(self):
        super().__init__()

    def calculate_embedding(self):
        return self.forward(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos,
                                                             neg)
        return loss

    def forward(self, all_users, all_items, graph):
        num_users = all_users.shape[0]
        num_items = all_items.shape[0]
        all_emb = torch.cat([all_users, all_items])
        embs = [all_emb]
        if DROPOUT and self.training:
            g_dropped = utils.dropout_x(graph, KEEP_PROB)
        else:
            g_dropped = graph
        for layer in range(N_LAYERS):
            all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        all_emb = torch.mean(embs, dim=1)
        return torch.split(all_emb, [num_users, num_items])
