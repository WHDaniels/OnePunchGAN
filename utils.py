from torchvision import transforms as ts
from nets import Generator, Discriminator
import torch


def get_augmented_images(image_A, image_B, mode):
    # Apply transformations:
    # horizontal flips --> 2x
    # rotating (non 90, 180, 270 degree) --> 4x
    # cropping (four quadrants) --> 4x
    # 2*4*4 = 32x original dataset

    if mode == 'train':

        augmentation = ts.Compose([
            ts.RandomHorizontalFlip(p=0.05),
            ts.RandomRotation((0, 360), fill=255),
            ts.Resize((image_A.size, image_A.size), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return {'A': augmentation(image_A),
                'B': augmentation(image_B)}

    else:

        augmentation = ts.Compose([
            # ts.RandomHorizontalFlip(p=0.05),
            # ts.RandomRotation((0, 360), fill=255),
            ts.Resize((image_B.size, image_B.size), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return {'A': augmentation(image_A),
                'B': augmentation(image_B)}


def make_directories():
    # make .train/A, .train/B
    # make .test/?
    # make .models/gen_A2B, .models/gen_B2A, .models/dis_A, .models/dis_B
    # make output

    return None


def initialize_nets(args, device):
    # initialize generators and discriminators
    gen_A2B = Generator().to(device)
    gen_B2A = Generator().to(device)
    dis_A = Discriminator().to(device)
    dis_B = Discriminator().to(device)

    # if generator or discriminator path specified, load them from directory
    if args.gen_A2B != 'empty':
        gen_A2B.load_state_dict(torch.load(args.gen_A2B))
    if args.gen_B2A != 'empty':
        gen_B2A.load_state_dict(torch.load(args.gen_B2A))
    if args.dis_A != 'empty':
        dis_A.load_state_dict(torch.load(args.dis_A))
    if args.dis_B != 'empty':
        dis_B.load_state_dict(torch.load(args.dis_B))

    return gen_A2B, gen_B2A, dis_A, dis_B
