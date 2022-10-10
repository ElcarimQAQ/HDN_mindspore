#Copyright 2022, Libing Yang

from mindspore import nn, ops
class TripletMarginLoss(nn.Cell):
    def __init__(self, margin=1.0, p=2):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p

    def calc_euclidean(self, x1, x2):
        # return (x1 - x2).pow(p).sum(1)
        return ops.norm(x1 - x2, -1, self.p)

    def construct(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        loss = ops.maximum(0.0, distance_positive - distance_negative + self.margin)
        return loss