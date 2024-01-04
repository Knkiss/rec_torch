import torch

import model
import world
from train import losses


class SimGCL(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.model = simgcl_encoder()

    def calculate_embedding(self):
        return self.model(False, self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)
        users_1, items_1 = self.model(True, self.embedding_user.weight, self.embedding_item.weight, self.Graph)
        users_2, items_2 = self.model(True, self.embedding_user.weight, self.embedding_item.weight, self.Graph)
        loss[losses.Loss.SSL.value] = losses.loss_info_nce(users_1, users_2, users) + \
                                      losses.loss_info_nce(items_1, items_2, pos)
        return loss


class simgcl_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.2

    def forward(self, perturbed, all_users, all_items, graph):
        num_users = all_users.shape[0]
        num_items = all_items.shape[0]
        all_emb = torch.cat([all_users, all_items])
        embs = []
        for layer in range(3):
            all_emb = torch.sparse.mm(graph, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb).to(world.device)
                all_emb += torch.sign(all_emb) * torch.nn.functional.normalize(random_noise, dim=-1) * self.eps
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        all_emb = torch.mean(embs, dim=1)
        return torch.split(all_emb, [num_users, num_items])
