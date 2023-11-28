import torch

import model
from train import losses
from train.losses import Loss


# MF Best result: {'Precision': array([0.02795559, 0.02009902]), 'NDCG': array([0.07912074, 0.0938307 ]), 'Recall': array([0.11618146, 0.16218443])}
# LightGCN Best result: {'Precision': array([0.0379776 , 0.02720044]), 'NDCG': array([0.11453868, 0.1359247 ]), 'Recall': array([0.17251831, 0.24137247])}

class EmbeddingBox(model.AbstractRecModel):
    def __init__(self):
        super().__init__()
        self.embedding_user_variance = torch.nn.Embedding(self.num_users, self.embedding_dim)
        self.embedding_item_variance = torch.nn.Embedding(self.num_items, self.embedding_dim)
        torch.nn.init.normal_(self.embedding_user_variance.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item_variance.weight, std=0.1)

        self.base_model = model.LightGCN()

    def calculate_loss(self, users, pos, neg):
        loss = {}
        users_mean, items_mean, user_var, items_var = self.calculate_embedding()

        loss[Loss.BPR.value] = losses.loss_BPR(users_mean, items_mean, users, pos, neg)
        loss[Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users, pos, neg)

        loss[Loss.SSL.value] = self.loss_variance(users_mean, items_mean, user_var, items_var, users, pos, neg)
        loss[Loss.Regulation.value] = losses.loss_regulation(self.embedding_user_variance, self.embedding_item_variance, users, pos, neg)
        return loss

    def loss_variance(self, users_mean, items_mean, user_var, items_var, users, pos, neg):
        users_emb_mean = users_mean[users.long()]
        pos_emb_mean = items_mean[pos.long()]
        neg_emb_mean = items_mean[neg.long()]

        users_emb_var = user_var[users.long()]
        pos_emb_var = items_var[pos.long()]
        neg_emb_var = items_var[neg.long()]

        users_pos_close, users_pos_far = self.getCloseAndFarEmbedding(users_emb_mean, pos_emb_mean, users_emb_var)
        items_pos_close, items_pos_far = self.getCloseAndFarEmbedding(pos_emb_mean, users_emb_mean, pos_emb_var)

        users_neg_close, users_neg_far = self.getCloseAndFarEmbedding(users_emb_mean, neg_emb_mean, users_emb_var)
        items_neg_close, items_neg_far = self.getCloseAndFarEmbedding(neg_emb_mean, users_emb_mean, neg_emb_var)

        # users_pos_close, users_pos_far = self.getCloseAndFarEmbedding(users_emb_mean.detach(), pos_emb_mean.detach(), users_emb_var)
        # items_pos_close, items_pos_far = self.getCloseAndFarEmbedding(pos_emb_mean.detach(), users_emb_mean.detach(), pos_emb_var)
        #
        # users_neg_close, users_neg_far = self.getCloseAndFarEmbedding(users_emb_mean.detach(), neg_emb_mean.detach(), users_emb_var)
        # items_neg_close, items_neg_far = self.getCloseAndFarEmbedding(neg_emb_mean.detach(), users_emb_mean.detach(), neg_emb_var)

        pos_list = []
        neg_list = []
        # TODO embedding的使用需要进一步设计
        # 每个节点的方差要尽可能大
        # neg_list.append(torch.mul(users_pos_close, users_pos_far).sum(dim=1))
        # neg_list.append(torch.mul(items_pos_close, items_pos_far).sum(dim=1))
        # neg_list.append(torch.mul(users_neg_close, users_neg_far).sum(dim=1))
        # neg_list.append(torch.mul(items_neg_close, items_neg_far).sum(dim=1))

        # 正样本之间的方差要尽可能大
        pos_list.append(torch.mul(users_pos_close, items_pos_far).sum(dim=1))
        pos_list.append(torch.mul(items_pos_close, users_pos_far).sum(dim=1))

        # 负样本之间的方差要尽可能小
        neg_list.append(torch.mul(users_neg_close, items_neg_far).sum(dim=1))
        neg_list.append(torch.mul(items_neg_close, users_neg_far).sum(dim=1))

        pos_scores = torch.stack(pos_list).sum(dim=0)
        neg_scores = torch.stack(neg_list).sum(dim=0)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        # loss = torch.sum(-torch.log(pos_scores / (neg_scores + pos_scores)))

        return loss

    def getCloseAndFarEmbedding(self, source_mean, target_mean, source_var):
        target_source_distance = target_mean - source_mean
        target_source_distance_norm = target_source_distance
        # TODO norm的意义是什么
        # target_source_distance_norm = torch.div(target_source_distance,
        #                                         torch.norm(target_source_distance, p=2, dim=1).unsqueeze(dim=1))
        source_target_diff = torch.mul(target_source_distance_norm, source_var)
        source_target_close = source_mean + source_target_diff
        source_target_far = source_mean - source_target_diff

        source_target_close = torch.nn.functional.normalize(source_target_close, dim=1)
        source_target_far = torch.nn.functional.normalize(source_target_far, dim=1)
        return source_target_close, source_target_far

    def getUsersRating(self, users):
        # 每epoch可能运行一次
        # 计算batch用户的得分
        users_mean, items_mean, users_var, items_var = self.calculate_embedding()
        users_mean = users_mean[users.long()]
        users_var = users_var[users.long()]

        distance = (items_mean.unsqueeze(dim=0) - users_mean.unsqueeze(dim=1))
        # TODO norm的意义是什么
        # diff = distance / torch.norm(distance, p=2, dim=2).unsqueeze(dim=2)
        diff = distance
        user_embs = users_mean.unsqueeze(dim=1) + torch.mul(diff, users_var.unsqueeze(dim=1))
        items_embs = items_mean.unsqueeze(dim=0) - torch.mul(diff, items_var.unsqueeze(dim=0))

        scores = torch.sum(torch.mul(user_embs, items_embs), dim=2)
        rating = self.f(scores)
        return rating

    def calculate_embedding(self):
        all_users_mean, all_items_mean = self.base_model(self.embedding_user.weight, self.embedding_item.weight, self.Graph)
        all_users_var = self.f(self.embedding_user_variance.weight)
        all_items_var = self.f(self.embedding_item_variance.weight)
        return all_users_mean, all_items_mean, all_users_var, all_items_var
