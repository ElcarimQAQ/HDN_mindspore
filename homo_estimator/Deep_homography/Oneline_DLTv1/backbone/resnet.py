import math

import mindspore.nn as nn
import torch.utils.model_zoo as model_zoo
import torch, imageio
import mindspore as ms
from mindspore import ops, Tensor, context
from mindspore.common.initializer import Normal, initializer,One
import mindspore_hub as mshub
# from utils import transform, DLT_solve
import matplotlib.pyplot as plt

"""
homo-estimator's backbone, reconstruction of the original Deephomography 
"""

criterion_l2 = nn.MSELoss()
# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False, size_average=False)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': "mindspore/1.8/resnet18_imagenet2012",
    'resnet34': "mindspore/1.8/resnet34_imagenet2012",
    'resnet50': "mindspore/1.8/resnet50_imagenet2012",
    'resnet101': "mindspore/1.8/resnet101_imagenet2012",
    'resnet152': "mindspore/1.8/resnet152_imagenet2012",
}


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def getPatchFromFullimg(patch_size_h, patch_size_w, patchIndices, batch_indices_tensor, img_full):
    num_batch, num_channels, height, width = img_full.size()
    warped_images_flat = img_full.reshape(-1)
    patch_indices_flat = patchIndices.reshape(-1)
    pixel_indices = patch_indices_flat.long() + batch_indices_tensor
    mask_patch = torch.gather(warped_images_flat, 0, pixel_indices)
    mask_patch = mask_patch.reshape([num_batch, 1, patch_size_h, patch_size_w])

    return mask_patch


def normMask(mask, strenth=0.5):
    """
    :return: to attention more region

    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask / (max_value * strenth)
    mask = ops.clip_by_value(mask, 0, 1)

    return mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, pad_mode='pad')


class BasicBlock(nn.Cell):#for resnet34, 18
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride) # p=1
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)# p=1
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):#
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# define and forward ( Because of the load is unbalanced when use torch.nn.DataParallel, we define warp in forward)
class ResNet(nn.Cell):

    def __init__(self, block,  layers, used_layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.used_layers = used_layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        print('block.expansion', block.expansion)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.cells():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight = initializer(Normal(0, math.sqrt(2. / n)), m.shape, ms.float32)
            if isinstance(m, nn.BatchNorm2d):
                fill = ops.Fill()
                m.weight = fill(ms.float32, (64,), 1)
                zeroslike = ops.ZerosLike()
                m.bias = zeroslike(Tensor(shape=(64,), dtype=ms.float32, init=One()))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)


    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        out = [x, p1, p2, p3, p4]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE)
    net = resnet34(used_layers=[2, 3, 4])
    print(net)

    template_var = ms.Tensor(shape=(1, 2, 127, 127), dtype=ms.float32, init=One())
    pretrained_dict = ms.load_checkpoint("/home/lbyang/workspace/HDN_mindspore/model/resnet34_ascend_v180_imagenet2012_official_cv_top1acc73.61_top5acc91.74.ckpt")
    ms.load_param_into_net(net, pretrained_dict)

    t = net(template_var)

    print(t[-1].shape)
    print(t[-1])


