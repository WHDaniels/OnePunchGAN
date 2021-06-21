import time

from torchvision import transforms as ts
from nets import Generator, Discriminator
import torch.nn as nn
import torch
import os


def get_augmented_images(image_A, image_B, mode, new_size):
    if mode == 'train':

        augmentation = ts.Compose([
            ts.RandomHorizontalFlip(p=0.05),
            ts.RandomRotation((0, 360), fill=255),
            ts.Resize((new_size, new_size), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return {'A': augmentation(image_A),
                'B': augmentation(image_B)}

    else:

        augmentation = ts.Compose([
            # ts.RandomHorizontalFlip(p=0.05),
            # ts.RandomRotation((0, 360), fill=255),
            ts.Resize((new_size, new_size), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return {'A': augmentation(image_A),
                'B': augmentation(image_B)}


def make_train_directories(args):
    # make train_A and train_B directories
    try:
        os.mkdir(args.train_path)
        os.mkdir(args.train_path + "\\train_A")
        os.mkdir(args.train_path + "\\train_B")
        print(f"Training path not found. Making at {args.train_path}")

    except FileExistsError as e:
        print(f"Training path found: {args.train_path}")

    # make models directory
    try:
        os.mkdir(args.save_path)
        print(f"Models path not found. Making at {args.save_path}")
    except FileExistsError as e:
        print(f"Models path found: {args.save_path}")


def make_test_directories(args):
    # make test directory
    try:
        os.mkdir(args.test_path)
        print(f"Testing path not found. Making at {args.test_path}")
    except FileExistsError as e:
        print(f"Training path found: {args.test_path}")

    # make output
    try:
        os.mkdir(args.output_path)
        print(f"Results path not found. Making at {args.output_path}")
    except FileExistsError as e:
        print(f"Results path found: {args.output_path}")


def initialize_nets(args, device):
    # initialize generators and discriminators
    gen_A2B = Generator().to(device)
    gen_B2A = Generator().to(device)
    dis_A = Discriminator().to(device)
    dis_B = Discriminator().to(device)

    # initialize weights to nets
    gen_A2B.apply(weights_init)
    gen_B2A.apply(weights_init)
    dis_A.apply(weights_init)
    dis_B.apply(weights_init)

    # if generator or discriminator path specified, load them from directory
    try:
        gen_A2B.load_state_dict(torch.load(args.gen_A2B))
        gen_B2A.load_state_dict(torch.load(args.gen_B2A))
        dis_A.load_state_dict(torch.load(args.dis_A))
        dis_B.load_state_dict(torch.load(args.dis_B))
    except FileNotFoundError as e:
        pass

    return gen_A2B, gen_B2A, dis_A, dis_B


# custom weights initialization called on generator and discriminator
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def move():
    import os
    import shutil

    for root, dirs, files in os.walk('E:\\One Punch Man'):  # replace the . with your starting directory
        for i, file in enumerate(files):
            path_file = os.path.join(root, file)
            shutil.copy2(path_file, 'C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\OnePunchGAN\\train\\train_B')
            new_path = os.path.join('C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\OnePunchGAN\\train\\train_B', file)
            renamed_path = os.path.join('C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\OnePunchGAN\\train\\train_B\\' +
                                        root[3:].replace('\\', '_') + '_' + str(i) + ".png")
            os.rename(new_path, renamed_path)
