import torch
import torch.nn.functional as F
from torch import nn

from util import utils
import world


class LightGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = world.lightGCN_layers
        self.keep_prob = world.lightGCN_keep_prob
        self.dropout = world.lightGCN_dropout

    def forward(self, all_users, all_items, graph):
        num_users = all_users.shape[0]
        num_items = all_items.shape[0]
        all_emb = torch.cat([all_users, all_items])
        embs = [all_emb]
        if self.dropout and self.training:
            g_droped = utils.dropout_x(graph, self.keep_prob)
        else:
            g_droped = graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        all_emb = torch.mean(embs, dim=1)
        return torch.split(all_emb, [num_users, num_items])


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2 * out_features, out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num

        # N, e_num, dim
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        We = entity_embs
        a_input = torch.cat((Wh, We), dim=-1)  # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), relations).sum(-1)  # N,e
        e = self.leakyrelu(e_input)  # (N, e_num)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs,
                          self.W)  # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We)  # (N, e_num, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N, e_num)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size())  # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1)  # (N, e_num, 2*out_features)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class QGrouping(nn.Module):
    def __init__(self):
        super(QGrouping, self).__init__()
        self.q_dim = 4          # 分组数量，结果embedding多少个组
        self.k_dim = 64         # 中间计算维度
        self.v_dim = 64         # 值维度，结果embedding多少个维度
        self.latent_dim = 64    # 输入维度
        self.Q = nn.Parameter(torch.Tensor(self.k_dim, self.q_dim))         # [k,q]
        self.W_K = nn.Parameter(torch.Tensor(self.latent_dim, self.k_dim))  # [d,k]
        self.W_V = nn.Parameter(torch.Tensor(self.latent_dim, self.v_dim))  # [d,v]
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
