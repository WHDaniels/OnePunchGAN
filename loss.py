import torch
import torch.nn as nn
from collections import OrderedDict


class CustomFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layers = [0, 1]
        self.under_layers = [4, 5, 6, 7]
        self.wgts = [1, 1, 1, 1]
        self.selected_out = OrderedDict()

        # PRETRAINED MODEL
        self.pretrained = torch.hub.load('RF5/danbooru-pretrained', 'resnet50').eval().to('cuda:1').requires_grad_(False)
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
        return [self.selected_out['0'],
                self.selected_out['1']]

    def forward(self, fake, target_rgb, target_greyscale=None):

        # fake_to_rgb = fake.tile((3, 1, 1))

        # need to translate greyscale image into 3 dimensional image to run through the pretrained resnet
        train_pass = self.pretrained(fake)
        train_act = self.get_activations()
        del train_pass

        target_pass = self.pretrained(target_rgb)
        target_act = self.get_activations()
        del target_pass

        # L1 loss is based off of the pair
        # image_loss = self.loss(fake, target_rgb)

        print(len(train_act))
        # feature loss is based off of the rgb pair
        feature_loss = sum([(self.loss(train_act[n], target_act[n]) * self.wgts[n])
                            for n, _ in enumerate(train_act)])  # + image_loss

        # return feature_loss
        return feature_loss


class CustomFeatureLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layers = ['0.4', '0.5', '0.6', '0.7']
        self.wgts = [7.5, 15, 1, 1]
        self.selected_out = OrderedDict()

        # PRETRAINED MODEL
        self.pretrained = torch.hub.load('RF5/danbooru-pretrained', 'resnet50').eval().to('cuda:1').requires_grad_(False)

        for name, module in self.pretrained.named_modules():
            if name in self.output_layers:
                module.register_forward_hook(self.forward_hook(name))

        self.loss = nn.L1Loss()

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def get_activations(self):
        return [self.selected_out[name] for name in self.output_layers]

    def forward(self, fake, target_rgb, target_greyscale=None):

        # fake_to_rgb = fake.tile((3, 1, 1))

        # need to translate greyscale image into 3 dimensional image to run through the pretrained resnet
        train_pass = self.pretrained(fake)
        train_act = self.get_activations()
        del train_pass

        target_pass = self.pretrained(target_rgb)
        target_act = self.get_activations()
        del target_pass

        # L1 loss is based off of the pair
        # image_loss = self.loss(fake, target_rgb)

        # feature loss is based off of the rgb pair
        feature_loss = sum([(self.loss(train_act[n], target_act[n]) * self.wgts[n])
                            for n, _ in enumerate(train_act)])  # + image_loss

        return feature_loss

