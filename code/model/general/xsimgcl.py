import torch

import model
import world
from train import losses

hyper_layers = 3  # l层GCN
hyper_layers_cl = 3  # l*层用于对比学习
hyper_eps = 0.2  # 对比学习的扰动
hyper_lambda = 0.2  # SSL的正则化参数
hyper_tau = 0.1  # SSL的温度参数

world.hyper_ssl_reg = hyper_lambda
world.hyper_ssl_temp = hyper_tau


class XSimGCL(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.model = xsimgcl_encoder(hyper_layers_cl, hyper_eps, hyper_layers)

    def calculate_embedding(self):
        return self.model(False, self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        users_all, items_1_all, users_cl, items_cl = self.model(True, self.embedding_user.weight,
                                                                self.embedding_item.weight, self.Graph)
        loss[losses.Loss.SSL.value] = losses.loss_info_nce(users_all, users_cl, users) + \
                                      losses.loss_info_nce(items_1_all, items_cl, pos)
        return loss


class xsimgcl_encoder(torch.nn.Module):
    def __init__(self, layer_cl, eps, layers):
        super().__init__()
        self.eps = eps
        self.layer_cl = layer_cl
        self.layers = layers

    def forward(self, perturbed, all_users, all_items, graph):
        num_users = all_users.shape[0]
        num_items = all_items.shape[0]
        all_emb = torch.cat([all_users, all_items])
        embs = []
        embs_cl = all_emb  # 某一层用于对比学习的emb
        for layer in range(self.layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb).to(world.device)
                all_emb += torch.sign(all_emb) * torch.nn.functional.normalize(random_noise, dim=-1) * self.eps
            embs.append(all_emb)
            if layer == self.layer_cl:
                embs_cl = all_emb
        embs = torch.stack(embs, dim=1)
        all_emb = torch.mean(embs, dim=1)
        user_all, item_all = torch.split(all_emb, [num_users, num_items])
        user_cl, item_cl = torch.split(embs_cl, [num_users, num_items])
        if perturbed:
            return user_all, item_all, user_cl, item_cl
        return user_all, item_all

