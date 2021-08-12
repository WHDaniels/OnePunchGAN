from utils import *
from torch.utils.data import Dataset
from torchvision import transforms as ts
from PIL import Image
import os


class PanelDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if self.args.mode == 'train':
            self.data_path_A = self.args.train_path + "\\train_A_old"
            self.data_path_B = self.args.train_path + "\\train_B_old"

        if self.args.mode == 'test':
            self.data_path_A, self.data_path_B = self.args.test_path, self.args.test_path

        self.image_list_A = list()
        self.image_list_B = list()

        self.get_image_paths()

        self.num_samples = self.args.critic_iters + 1

    def __getitem__(self, i):
        transform_seed = (random.randint(0, 2 ** 32))

        self.image_path_A = os.path.join(self.data_path_A, self.image_list_A[i])
        self.image_path_B = os.path.join(self.data_path_B, self.image_list_B[i])

        self.image_A = Image.open(self.image_path_A).convert('RGB')
        self.image_B = Image.open(self.image_path_B).convert('RGB')

        min_dims = (min(self.image_A.size[0], self.image_B.size[0]),
                    min(self.image_A.size[1], self.image_B.size[1]))

        image_A_rgb, image_A_greyscale = self.get_transform(self.image_A, min_dims, transform_seed)
        image_B_rgb, image_B_greyscale = self.get_transform(self.image_B, min_dims, transform_seed)

        return {'A_rgb': image_A_rgb, 'A_greyscale': image_A_greyscale,
                'B_rgb': image_B_rgb, 'B_greyscale': image_B_greyscale}

    def __len__(self):
        return len(self.image_list_A) // self.args.batch_size

    def get_image_paths(self):
        for root, dirs, files in os.walk(self.data_path_A):
            for name in files:
                self.image_list_A.append(os.path.join(root, name))

        for root, dirs, files in os.walk(self.data_path_B):
            for name in files:
                self.image_list_B.append(os.path.join(root, name))

    def get_transform(self, image, min_dims, transform_seed):
        compose_list_rgb = list()
        compose_list_greyscale = list()

        if self.args.mode == 'train':

            self.set_seeds(transform_seed)
            compose_list_rgb += [ts.Resize(min_dims, interpolation=ts.InterpolationMode.BICUBIC)]
            self.set_seeds(transform_seed)
            compose_list_greyscale += [ts.Resize(min_dims, interpolation=ts.InterpolationMode.BICUBIC)]

            scale = random.uniform(2, 10) / 10
            crop_area = (int(min_dims[0] * scale), int(min_dims[1] * scale))

            compose_list_rgb += [
                ts.RandomCrop(crop_area),
                ts.RandomHorizontalFlip(p=0.5),
                ts.RandomRotation(25, fill=255),
                ts.RandomPerspective(distortion_scale=0.15, p=0.33, fill=255),

                ts.Resize((self.args.image_size, self.args.image_size), interpolation=ts.InterpolationMode.BICUBIC),
                ts.ToTensor(),
                ts.RandomErasing(p=0.5, scale=(0.005, 0.1)),
                ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

            compose_list_greyscale += [
                ts.RandomCrop(crop_area),
                ts.RandomHorizontalFlip(p=0.5),
                ts.RandomRotation(25, fill=255),
                ts.RandomPerspective(distortion_scale=0.15, p=0.33, fill=255),

                ts.Grayscale(num_output_channels=1),
                ts.Resize((self.args.image_size, self.args.image_size), interpolation=ts.InterpolationMode.BICUBIC),
                ts.ToTensor(),
                ts.RandomErasing(p=0.5, scale=(0.005, 0.1)),
                ts.Normalize((0.5,), (0.5,))
            ]

            self.set_seeds(transform_seed)
            rgb_transform = ts.Compose(compose_list_rgb)(image)
            self.set_seeds(transform_seed)
            greyscale_transform = ts.Compose(compose_list_greyscale)(image)

            return rgb_transform, greyscale_transform

        else:

            compose_list_rgb += [
                ts.Resize((self.args.image_size, self.args.image_size), interpolation=ts.InterpolationMode.BICUBIC),
                ts.ToTensor(),
                ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

            return ts.Compose(compose_list_rgb)

    @staticmethod
    def set_seeds(transform_seed):
        random.seed(transform_seed)
        torch.manual_seed(transform_seed)
        torch.cuda.manual_seed(transform_seed)
        np.random.seed(transform_seed)
        torch.cuda.manual_seed_all(transform_seed)