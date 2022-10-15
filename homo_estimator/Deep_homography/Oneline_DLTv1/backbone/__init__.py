from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mindspore import nn
import homo_estimator.Deep_homography.Oneline_DLTv1.backbone.resnet as resnet
import  mindspore_hub as mshub
import mindspore as ms


model_urls = {
    'resnet34': "/home/lbyang/workspace/HDN_mindspore/pretrained_models/resnet34.ckpt", # "mindspore/1.8/resnet34_imagenet2012"
}

def get_backbone(model_name, pretrained=False, **kwargs):
    print('**kwargs',kwargs)
    if model_name == 'resnet34':
        model = resnet.resnet34(pretrained=False, **kwargs)
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')

    if pretrained == True:
        print('load_pretrained from',model_urls[model_name])
        exclude_dict = ['conv1.weight','fc.weight','fc.bias']
        pretrained_dict = ms.load_checkpoint(model_urls[model_name])
        model_dict = model.parameters_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in exclude_dict}

        model_dict.update(pretrained_dict)
        ms.load_param_into_net(model, model_dict)


    return model

