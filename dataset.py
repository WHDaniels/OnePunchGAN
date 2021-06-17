from utils import *
from torch.utils.data import Dataset
from torchvision import transforms as ts
from PIL import Image
import os


class PanelDataset(Dataset):
    def __init__(self, data_path_A, data_path_B, image_size, mode):
        self.mode = mode
        self.data_path_A = data_path_A
        self.data_path_B = data_path_B
        self.image_size = image_size
        self.image_list_A = os.listdir(self.data_path_A)
        self.image_list_B = os.listdir(self.data_path_B)

    def __getitem__(self, i):
        # Resize pictures
        self.image_path_A = os.path.join(self.data_path_A, self.image_list_A[i])
        self.image_path_B = os.path.join(self.data_path_B, self.image_list_B[i])

        self.image_A = Image.open(self.image_path_A)
        self.image_B = Image.open(self.image_path_B)

        self.original_B_size = self.image_B.size  # may not need
        self.original_A_size = self.image_A.size  # may not need

        # maybe see here if you can crop a 512x512 image out instead of resizing (image would need to be very big)
        # --> refactor above if not implemented

        return get_augmented_images(self.image_A, self.image_B, self.mode)

    def __len__(self):
        return len(self.image_list_A)
