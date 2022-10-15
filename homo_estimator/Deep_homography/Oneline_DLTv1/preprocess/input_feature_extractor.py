import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import  HeNormal

class PreShareFeature(nn.Cell):
    def __init__(self, ):
        super(PreShareFeature, self).__init__()
        self.ShareFeature = nn.SequentialCell(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, pad_mode='pad',weight_init=HeNormal()),
            nn.BatchNorm2d(4, momentum=0.9),
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, pad_mode='pad',weight_init=HeNormal()),
            nn.BatchNorm2d(8, momentum=0.9),
            nn.ReLU(),

            nn.Conv2d(8, 1, kernel_size=3, padding=1, pad_mode='pad',weight_init=HeNormal()),
            nn.BatchNorm2d(1, momentum=0.9),
            nn.ReLU(),
        )
        print('ShareFeature.param', self.ShareFeature[0].weight)
        # for m in self.cells_and_names():
        #     if isinstance(m[1], nn.Conv2d):
        #         # nn.init.kaiming_normal_(m.weight)
        #         m[1].weight_init = HeNormal
        #     elif isinstance(m[1], nn.BatchNorm2d):
        #         # m.weight.data.fill_(1)
        #         # m.bias.data.zero_()
        #         m[1].gamma.data.fill(1)
        #         ops.ZerosLike()(m[1].beta.data)

    def construct(self, x):
        out = self.ShareFeature(x)
        return out


