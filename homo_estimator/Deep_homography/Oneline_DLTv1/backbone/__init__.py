from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mindspore import nn
import homo_estimator.Deep_homography.Oneline_DLTv1.backbone.resnet as resnet
import  mindspore_hub as mshub
import mindspore as ms
# from test_ideas.net.unet import  UNet_fuse



model_urls = {
    'resnet18': "mindspore/1.8/resnet18_imagenet2012",
    'resnet34': "mindspore/1.8/resnet34_imagenet2012",
    'resnet50': "mindspore/1.8/resnet50_imagenet2012",
    'resnet101': "mindspore/1.8/resnet101_imagenet2012",
    'resnet152': "mindspore/1.8/resnet152_imagenet2012",
}

def get_backbone(model_name, pretrained=False, **kwargs):
    print('**kwargs',kwargs)
    if model_name == 'resnet34':
        model = resnet.resnet34(pretrained=False, **kwargs)
    elif model_name == 'resnet50':
        model = resnet.resnet50(pretrained=False, **kwargs)
    elif model_name == 'resnet101':
        model = resnet.resnet101(pretrained=False, **kwargs)
    elif model_name == 'resnet152':
        model = resnet.resnet152(pretrained=False, **kwargs)

    if model_name == 'resnet18':
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
    elif model_name == 'resnet34':
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
    else:
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
    if pretrained == True:
        print('load_pretrained from',model_urls[model_name])
        exclude_dict = ['conv1.weight','fc.weight','fc.bias']
        pretrained_dict = ms.load_checkpoint("/home/lbyang/workspace/HDN_mindspore/model/resnet34_ascend_v180_imagenet2012_official_cv_top1acc73.61_top5acc91.74.ckpt")
        model_dict = model.parameters_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in exclude_dict}

        model_dict.update(pretrained_dict)
        ms.load_param_into_net(model, model_dict)


    return model

# def get_backbone_unet(name, **kwargs):
#     if name == 'Unet_fuse':
#         return UNet_fuse(**kwargs)
#     return UNet_fuse(**kwargs)