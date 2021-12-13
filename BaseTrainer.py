from main.dataset import PanelDataset, FinalDataset
from torch.utils.data import DataLoader
from time import perf_counter
from PIL import Image
import numpy as np
import torch
import os

torch.backends.cudnn.benchmark = True


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.channels = 3

        if self.args.model == 'gen':
            self.device1 = torch.device('cuda:1')
            if self.args.multi_gpu:
                self.device2 = torch.device('cuda:1')
        else:
            self.device1 = torch.device('cuda:0')
            self.device2 = torch.device('cuda:0')

        self.loss_list, self.total_loss_list = [], []
        self.gen_A2B_loss_list, self.gen_B2A_loss_list = [], []
        self.dis_A_loss_list, self.dis_B_loss_list = [], []
        self.avg_gen_loss, self.avg_dis_loss = [], []

        # creating targets
        # t_size = int((self.args.image_size / 64) ** 2)
        t_size = 14

        self.target_real1 = torch.full((self.args.batch_size, 1, t_size, t_size), 1, device=self.device1,
                                       dtype=torch.float32)
        self.target_fake1 = torch.full((self.args.batch_size, 1, t_size, t_size), 0, device=self.device1,
                                       dtype=torch.float32)

        if self.args.multi_gpu:
            self.target_real2 = torch.full((self.args.batch_size, 1, t_size, t_size), 1, device=self.device2,
                                           dtype=torch.float32)
            self.target_fake2 = torch.full((self.args.batch_size, 1, t_size, t_size), 0, device=self.device2,
                                           dtype=torch.float32)

        if 'final' in self.args.model:
            self.dataset = FinalDataset(self.args)
        else:
            self.dataset = PanelDataset(self.args)

        if self.args.mode == 'train':
            if self.args.model == 'gan':
                self.loader = DataLoader(self.dataset, self.args.batch_size * self.args.critic_iters + 1,
                                         shuffle=self.args.shuffle, pin_memory=True, drop_last=True)
            else:
                self.loader = DataLoader(self.dataset, self.args.batch_size, shuffle=self.args.shuffle,
                                         pin_memory=True, drop_last=True, num_workers=1)

        else:
            self.loader = DataLoader(self.dataset, args.batch_size)

    def initialize_training(self):

        print("Making directories...")
        self.make_train_directories()

        print("Initializing generators and discriminators...")
        self.initialize_nets()

        print("Loading network weights...")
        self.load_weights()

        print("Defining optimizers...")
        self.define_optimizers()

        print("Defining schedulers...")
        self.define_schedulers()

    def make_train_directories(self):
        # make train_A and train_B directories
        try:
            os.mkdir(self.args.train_path)
            os.mkdir(self.args.folder_A)
            os.mkdir(self.args.folder_B)
            print(f"Training path not found. Making at {self.args.train_path}")

        except FileExistsError:
            print(f"Training path found: {self.args.train_path}")

        # make models directory
        try:
            os.mkdir(self.args.save_path)
            print(f"Models path not found. Making at {self.args.save_path}")
        except FileExistsError:
            print(f"Models path found: {self.args.save_path}")

    def make_test_directories(self):
        # make test directory
        try:
            os.mkdir(self.args.test_path)
            print(f"Testing path not found. Making at {self.args.test_path}")
        except FileExistsError:
            print(f"Test path found: {self.args.test_path}")

        # make output
        try:
            os.mkdir(self.args.output_path)
            print(f"Results path not found. Making at {self.args.output_path}")
        except FileExistsError:
            print(f"Results path found: {self.args.output_path}")

    def gather_losses(self, i):
        """
        Gathers losses of all models.
        :param i: Current iteration over epoch.
        :return: List of losses from all models.
        """

        if not self.total_loss_list:
            self.total_loss_list = [loss[1] for loss in self.loss_list]
        else:
            for n in range(len(self.total_loss_list)):
                self.total_loss_list[n] += self.loss_list[n][1]

        if i + 1 == len(self.loader):
            for n in range(len(self.total_loss_list)):
                self.total_loss_list[n] /= len(self.loader)

    def give_training_eta(self, start, i, current_epoch):
        # Give a training ETA
        end = perf_counter()

        if i % 75 == 0:
            seconds_per_batch = end - start
            print('-' * 50)
            print(f"\nTime per batch: {seconds_per_batch} seconds")

            if self.args.model == 'gan':
                hours_per_epoch = seconds_per_batch * len(self.dataset) / self.args.critic_iters / 3600
            else:
                hours_per_epoch = seconds_per_batch * len(self.dataset) / 3600
            print(f"Time per epoch: {hours_per_epoch} hours")

            hours_per_train = hours_per_epoch * (self.args.epochs + self.args.decay_epochs)
            print(f"Time per training: {hours_per_train / 24} days\n")

            hours_left = hours_per_train - ((current_epoch) * hours_per_epoch + (i * seconds_per_batch / 3600))
            print("Hours left:", hours_left)
            print("Days left:", hours_left / 24)
            print('-' * 50)

    @staticmethod
    def save_images(image_tensor_dict, args, epoch, i):

        for image_tensor_item in image_tensor_dict.items():
            image_tensor = image_tensor_item[1].data
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            # post-processing: transpose and scaling
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            image = Image.fromarray(image_numpy.astype(np.uint8))
            image.save(f'{args.results_path}\\epoch_{epoch}\\{i}_{image_tensor_item[0]}.png')

    def initialize_nets(self):
        """
        Initialize all networks relevant to subclass model architecture. Should override this function.
        :return: None
        """

    def load_weights(self):
        """
        Load all network weights relevant to subclass model architecture. Should override this function.
        :return: None
        """

    def define_optimizers(self):
        """
        Define all optimizers relevant to subclass model architecture. Should override this function.
        :return: None
        """

    def define_schedulers(self):
        """
        Define all schedulers relevant to subclass model architecture. Should override this function.
        :return: None
        """

    def get_results(self, i, epoch, real_A1, real_B1):
        """
        Show results after a multiple of iterations. Should override this function.
        """

    def save_training(self, epoch):
        """
        Update schedulers, save state_dict of networks, and plot loss metrics. Should override this function.
        """
        pass

    def final_save(self):
        """
         Save final models. Should override this function.
        """
        pass

    def test(self):
        """
        Test inputs to trained model. Should override this function.
        """
        pass
