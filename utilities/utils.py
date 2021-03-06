from torch.nn.init import xavier_uniform_
import torchvision.utils as vutils
from torch.nn import init
import torch.nn as nn
import numpy as np
import torch
import os


# init_weights for CycleGAN generator and discriminator
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# custom weights initialization for custom generator and critic
def weights_init(m):
    # class_name = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.InstanceNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_disc_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def move():
    import shutil

    for root, dirs, files in os.walk('E:\\One Punch Man'):
        for i, file in enumerate(files):
            path_file = os.path.join(root, file)
            shutil.copy2(path_file, 'C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\OnePunchGAN\\train\\train_B')
            new_path = os.path.join('C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\OnePunchGAN\\train\\train_B', file)
            renamed_path = os.path.join('C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\OnePunchGAN\\train\\train_B\\' +
                                        root[3:].replace('\\', '_') + '_' + str(i) + ".png")
            os.rename(new_path, renamed_path)


def save_images(image_tensor_dict, args, epoch, i):
    for image_tensor_item in image_tensor_dict.items():
        image_tensor = image_tensor_item[1].data
        if image_tensor_item[0] == 'almost_real_A':
            image = UnNormalize([0.7137, 0.6628, 0.6519], [0.2970, 0.3017, 0.2979])(image_tensor)
        else:
            image = UnNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor)

        vutils.save_image(image.detach(), f'{args.results_path}\\epoch_{epoch}\\{i}_{image_tensor_item[0]}.png')


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


def denormalize(array, mean, std):
    array = np.array([array[:, :, n] * std[n] + mean[n] for n in range(0, 3)])
    array = np.transpose(array, (1, 2, 0))
    return array


def to_greyscale(tensor):
    r_tensor, g_tensor, b_tensor = tensor[0][0] * 0.3, tensor[0][1] * 0.59, tensor[0][2] * 0.11
    grey_tensor = r_tensor + g_tensor + b_tensor
    return grey_tensor


def changeBN2IN(model_path):
    model = torch.load(model_path)
    new_model = batch_norm_to_group_norm(model)

    for param in new_model.named_parameters():
        print(param)

    torch.save(new_model, './pretrained/inst_final_resnet_50_CondInst.pth')


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    layer._modules[name] = torch.nn.InstanceNorm2d(sub_layer.num_features, affine=True,
                                                                   track_running_stats=True)

            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                batch_norm_to_group_norm(sub_layer)
    return layer

# changeBN2IN("pretrained/final_resnet_50_CondInst.pth")
