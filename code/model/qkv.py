import torch
from torch import nn

import model
from train import losses

Q_GROUP = 4
K_DIM = 64
V_DIM = 64
LATENT_DIM = 64
QKV_ONLY_USER = False


class QGrouping(nn.Module):
    def __init__(self):
        super(QGrouping, self).__init__()
        self.Q = nn.Parameter(torch.Tensor(K_DIM, Q_GROUP))       # [k,q]
        self.W_K = nn.Parameter(torch.Tensor(LATENT_DIM, K_DIM))  # [d,k]
        self.W_V = nn.Parameter(torch.Tensor(LATENT_DIM, V_DIM))  # [d,v]
        nn.init.xavier_uniform_(self.Q, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_K, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_V, gain=nn.init.calculate_gain('relu'))

    def forward(self, emb):
        K = torch.matmul(emb, self.W_K).unsqueeze(1)      # [N,1,k]
        V = torch.matmul(emb, self.W_V).unsqueeze(2)      # [N,v,1]
        att = torch.matmul(K, self.Q)                     # [N,1,q]
        att = torch.softmax(att, dim=2)
        final = torch.matmul(V, att)                      # [N,v,q]
        return final

class QKV(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.lightGCN = model.LightGCN()
        self.QGrouping = QGrouping()

    def calculate_embedding(self):
        return self.forward(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        return loss

    def forward(self, all_users, all_items, graph):
        all_users, all_items = self.lightGCN(all_users, all_items, graph)
        if not QKV_ONLY_USER:
            all_users, all_items = self.QGrouping(all_users), self.QGrouping(all_items)
            all_users = torch.reshape(all_users, shape=[all_users.shape[0], V_DIM * Q_GROUP])
            all_items = torch.reshape(all_items, shape=[all_items.shape[0], V_DIM * Q_GROUP])
        else:
            all_users = self.QGrouping(all_users)
            all_users = torch.mean(all_users, dim=2)
        return all_users, all_items
