import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def mcncc(self, v1, v2):
        batch_size = v1.shape[0]
        den = torch.norm(v1, dim=2)*torch.norm(v2, dim=2) + 1e-12
        num = (v1*v2).sum(dim=2)
        ncc = num/den
        ncc_mean = ncc.mean(dim=1)
        return ncc_mean

    def __init__(self, margin, ncc=False):
        super(TripletLoss, self).__init__()
        self.ncc = ncc
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        if not self.ncc:
            distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
            # losses = F.relu(distance_positive - distance_negative + self.margin)
            ranking_loss = nn.MarginRankingLoss(margin=self.margin)
            y = Variable(distance_negative.data.new().resize_as_(distance_negative.data).fill_(1))
            losses = ranking_loss(distance_negative, distance_positive, y)
            return losses.mean() if size_average else losses.sum()
        else:
            # print("anchor.shape {}".format(anchor.shape))
            anchor_reshape = anchor.reshape(anchor.shape[0], anchor.shape[1], anchor.shape[2]*anchor.shape[3])
            negative_reshape = negative.reshape(negative.shape[0], negative.shape[1], negative.shape[2]*negative.shape[3])
            positive_reshape = positive.reshape(positive.shape[0], positive.shape[1], positive.shape[2]*positive.shape[3])
            # print("self.mcncc(anchor_reshape, negative_reshape) = {}".format(self.mcncc(anchor_reshape, negative_reshape)))
            losses =  self.mcncc(anchor_reshape, negative_reshape) \
                - self.mcncc(anchor_reshape, positive_reshape) + self.margin
            losses_relu = F.relu(losses)
            return losses_relu.mean()

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
