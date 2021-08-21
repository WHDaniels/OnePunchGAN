import os

import numpy
import torch
from time import perf_counter

import torchvision
import numpy as np
from PIL import Image

from image_pool import ImagePool
import torchvision.utils as vutils
from BaseTrainer import BaseTrainer
from matplotlib import pyplot as plt
from torchvision import transforms as ts

from loss import CustomFeatureLoss
from nets import ColorNet, Discriminator, Generator
from utils import save_images, weights_init, init_weights, denormalize


class FinalTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.adversarial_loss = torch.nn.MSELoss().to(self.device2)
        self.cycle_consistency_loss = torch.nn.L1Loss().to(self.device2)
        # self.feature_loss = CustomFeatureLoss()
        # self.identity_loss = torch.nn.L1Loss().to(self.device2)

        self.gen_A2B_optimizer, self.dis_B_optimizer = None, None
        self.gen_A2B_scheduler, self.dis_B_scheduler = None, None

        self.gen_A2B, self.gen_B2A = None, None
        self.dis_B = None
        self.identity_B = None

        self.fake_A, self.fake_B = None, None
        self.real_A, self.rec_B = None, None

        self.fake_B_pool = ImagePool(50)

    # Assuming multi_gpu for now
    def train(self):

        self.initialize_training()

        print("Starting training...")
        for epoch in range(0, self.args.epochs + self.args.decay_epochs):
            self.total_loss_list = list()

            for i, batch in enumerate(self.loader):
                print(f"Epoch {epoch} | Iter {i}")

                start = perf_counter()

                # just get real B
                real_B_reg_norm = batch['real_B_reg_norm'].to(self.device1)
                real_B_pre_norm = self.normalize_to_pretrained(real_B_reg_norm).to(self.device2)

                # Freeze discriminators
                self.dis_B.requires_grad_(False)

                # Train both generators
                gen_A2B_loss = self.train_generators(real_B_reg_norm, real_B_pre_norm)
                self.gen_A2B_loss_list.append(gen_A2B_loss.cpu().detach().numpy())

                # Train discriminator
                dis_B_loss = self.train_discriminator_B(real_B_pre_norm)
                self.dis_B_loss_list.append(dis_B_loss.cpu().detach().numpy())

                self.give_training_eta(start, i, epoch)
                # self.gather_losses(i)

                save_dict = self.get_results(i, epoch, self.real_A, real_B_pre_norm)
                save_images(save_dict, self.args, epoch, i)

            self.save_training(epoch)

        self.final_save()

    def train_generators(self, real_B1, real_B2):

        # Set generator gradient to zero
        self.gen_A2B_optimizer.zero_grad()

        # removing identity loss for now
        # self.identity_A = self.gen_B2A(real_A2)
        # identity_A_loss = self.identity_loss1(self.identity_A, real_A2_grey) * lambda_B * lambda_identity

        # self.identity_B = self.gen_A2B(real_B2)
        # identity_B_loss = self.identity_loss(self.identity_B, real_B2) * 0.5

        # Get "real" black and white input
        with torch.no_grad():
            self.real_A = self.gen_B2A(real_B1)
        self.real_A = self.normalize_to_pretrained().to(self.device2)

        # Getting adversarial loss
        self.fake_B = self.gen_A2B(self.real_A)

        gen_A2B_loss = self.adversarial_loss(self.dis_B(self.fake_B), self.target_real2)  # GAN loss D_A(G_A(A))

        # B input to cycle loss here is real B (colored image)
        cycle_BAB_loss = self.cycle_consistency_loss(self.fake_B, real_B2) * 10

        # perceptual_loss = self.feature_loss(self.fake_B, real_B2)

        # Combine loss
        gen_error = gen_A2B_loss + cycle_BAB_loss

        # Calculate gradients for both generators
        gen_error.backward()

        # Update weights of both generators
        self.gen_A2B_optimizer.step()

        # print("cycle_BAB_loss", cycle_BAB_loss.detach().cpu())
        # print("gen_A2B_loss", gen_A2B_loss.detach().cpu())
        # print("feature_loss", perceptual_loss.detach().cpu())

        return gen_error

    def train_discriminator_B(self, real):
        # Unfreeze discriminators
        self.dis_B.requires_grad_(True)

        # Set discriminator gradients to zero
        self.dis_B_optimizer.zero_grad()

        # Loss for discriminator A
        new_fake = self.fake_B_pool.query(self.fake_B)

        # Getting predictions
        real_prediction = self.dis_B(real)
        fake_prediction = self.dis_B(new_fake.detach())

        # Getting discriminator loss
        dis_real_loss = self.adversarial_loss(real_prediction, self.target_real2)
        dis_fake_loss = self.adversarial_loss(fake_prediction, self.target_fake2)

        # Calculating discriminator error
        dis_B_loss = (dis_real_loss + dis_fake_loss) * 0.5

        # Gradient update
        dis_B_loss.backward()

        # update discriminator weights
        self.dis_B_optimizer.step()

        return dis_B_loss

    def initialize_nets(self):
        if self.args.multi_gpu:
            self.gen_A2B = ColorNet(3).to(self.device2)

            rest = False
            for x, param in self.gen_A2B.named_parameters():
                if 'bridge.0.weight' in x:
                    rest = True
                if rest:
                    if 'weight' in x:
                        if 'norm' in x:
                            torch.nn.init.ones_(param)
                        else:
                            torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
                else:
                    param.requires_grad = False

            self.gen_B2A = Generator(3, 1).to(self.device1).requires_grad_(False)
            self.dis_B = Discriminator(3).to(self.device2)
            self.dis_B.apply(init_weights)

        else:
            self.gen_A2B = ColorNet().to(self.device1)
            self.gen_B2A = ColorNet().to(self.device1)
            self.dis_B = Discriminator(3).to(self.device1)
            self.dis_B.apply(init_weights)

    def load_weights(self):
        # if generator or discriminator path specified, load them from directory
        if self.args.gen_A2B_dict != '':
            self.gen_A2B.load_state_dict(torch.load(self.args.gen_A2B_dict))

        if self.args.gen_B2A_dict != '':
            self.gen_B2A.load_state_dict(torch.load(self.args.gen_B2A_dict))

        if self.args.dis_B_dict != '':
            self.dis_B.load_state_dict(torch.load(self.args.dis_B_dict))

    def define_optimizers(self):
        self.gen_A2B_optimizer = torch.optim.Adam(self.gen_A2B.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.dis_B_optimizer = torch.optim.Adam(self.dis_B.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

    def define_schedulers(self):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.args.epochs) / float(self.args.decay_epochs + 1)

        self.gen_A2B_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_A2B_optimizer, lr_lambda=lambda_rule)
        self.dis_B_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_B_optimizer, lr_lambda=lambda_rule)

    def get_results(self, i, epoch, real_A, real_B):
        """
        Show results after a multiple of iterations.
        """
        tensor_dict = {}

        if i % 250 == 0:
            # show results on test per epoch
            try:
                os.mkdir(os.getcwd() + f'\\results\\epoch_{epoch}')

            except FileExistsError:
                print("Directory exists already...")

            finally:
                # Save image
                tensor_dict = {
                    "almost_real_A": real_A,
                    "fake_B": self.fake_B,
                    "real_B": real_B
                }

        return tensor_dict

    def save_training(self, epoch):

        # update learning rates
        self.gen_A2B_scheduler.step()
        self.dis_B_scheduler.step()

        # checkpoints
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_epoch_{epoch}.pth")
        torch.save(self.dis_B.state_dict(), f"{self.args.save_path}/dis_B_epoch_{epoch}.pth")

        if self.args.metrics:
            # plot loss versus epochs
            plt.figure(figsize=[8, 6])
            plt.plot(self.gen_A2B_loss_list, 'r', linewidth=2, label='gen_A2B')
            plt.plot(self.dis_B_loss_list, 'c', linewidth=2, label='dis_B')
            plt.xlabel('Epochs', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.legend()
            plt.savefig(f"figures/epoch{epoch}.png")

    def final_save(self):
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_final.pth")
        torch.save(self.dis_B.state_dict(), f"{self.args.save_path}/dis_B_final.pth")

    def normalize_to_pretrained(self, opt_image=None):
        if opt_image is None:
            image_tensor = self.real_A.data
        else:
            image_tensor = opt_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = Image.fromarray(image_numpy.astype(np.uint8))

        gen_image = ts.ToTensor()(image)
        gen_image = ts.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979])(gen_image)
        return torch.unsqueeze(gen_image, dim=0)

    def test(self):
        print("Making directories...")
        self.make_test_directories()

        print("Initializing generators...")
        self.initialize_nets()

        print("Loading weights...")
        self.load_weights()

        print("Setting model mode...")
        self.gen_A2B.eval()
        self.gen_B2A.eval()

        for i, batch in enumerate(self.loader):
            # get batch data
            if self.args.multi_gpu:
                input_panel_A = batch['A'].to(self.device1)
                input_panel_B = batch['B'].to(self.device2)
            else:
                input_panel_A = batch['A'].to(self.device1)
                input_panel_B = batch['B'].to(self.device1)

            output_panel_A = 0.5 * (self.gen_A2B(input_panel_A).data + 1.0)
            output_panel_B = 0.5 * (self.gen_B2A(input_panel_B).data + 1.0)

            vutils.save_image(output_panel_A.detach(), f"{self.args.output_path}/A_{i}.png")
            vutils.save_image(output_panel_B.detach(), f"{self.args.output_path}/B_{i}.png")
