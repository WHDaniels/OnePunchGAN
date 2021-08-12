# Reference:
# https://medium.com/the-owl/using-forward-hooks-to-extract-intermediate-layer-outputs-from-a-pre-trained-model-in-pytorch-1ec17af78712

from skimage.color import gray2rgb, rgb2gray
import numpy as np
import torch
import torchvision
import torch.nn as nn
from fastai.callback.hook import hook_outputs
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.m_feat = torchvision.models.vgg16_bn(True).features.eval().requires_grad_(False)
        blocks = [
            i - 1
            for i, o in enumerate(self.m_feat.children())
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = [.2, .7, .1]
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = nn.L1Loss()

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


class CustomFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layers = [5, 6, 7]
        self.wgts = [20, 70, 10]
        self.selected_out = OrderedDict()

        # PRETRAINED MODEL
        # self.pretrained = generator
        self.pretrained = torchvision.models.resnet50(pretrained=True).eval().to('cuda:1').requires_grad_(False)
        # self.pretrained = torch.load('./pretrained/inst_final_resnet_50_CondInst.pth').eval().to('cuda:1').requires_grad_(False)

        self.fhooks = []
        self.loss = nn.L1Loss()

        for i, l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained, l).register_forward_hook(self.forward_hook(l)))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def get_activations(self):
        return [self.selected_out['layer2'],
                self.selected_out['layer3'],
                self.selected_out['layer4']]

    def forward(self, fake, target_rgb, target_greyscale):

        fake_to_rgb = fake.tile((3, 1, 1))

        # need to translate greyscale image into 3 dimensional image to run through the pretrained resnet
        train_pass = self.pretrained(fake_to_rgb)
        train_act = self.get_activations()
        del train_pass

        target_pass = self.pretrained(target_rgb)
        target_act = self.get_activations()
        del target_pass

        # L1 loss is based off of the greyscale pair
        image_loss = self.loss(fake, target_greyscale)

        # feature loss is based off of the rgb pair
        feature_loss = sum([(self.loss(train_act[n], target_act[n]) * self.wgts[n])
                            for n, _ in enumerate(train_act)]) + image_loss

        return feature_loss