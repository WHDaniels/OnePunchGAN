import os
from time import perf_counter
from torchvision import utils as vutils
import torch
from matplotlib import pyplot as plt
from BaseTrainer import BaseTrainer
from loss import CustomFeatureLoss
from nets import ColorNet
import matplotlib
from importlib import reload


class GenTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.output_dim = 1
        self.A2B = False

        self.gen_A2B = None
        self.gen_A2B_optimizer = None
        self.gen_A2B_scheduler = None
        self.fake_B = None

        self.loss = CustomFeatureLoss()

    def train(self):

        self.initialize_training()

        print("Starting training...")
        for epoch in range(1, self.args.epochs + self.args.decay_epochs):
            self.total_loss_list = list()

            for i, batch in enumerate(self.loader):
                print(f"Epoch {epoch} | Iter {i}")

                start = perf_counter()

                # get batch data
                if self.args.multi_gpu:
                    real_A_rgb = batch['A_rgb'].to(self.device1)
                    real_B_rgb = batch['B_rgb'].to(self.device2)
                    real_A_greyscale = batch['A_greyscale'].to(self.device1)
                    real_B_greyscale = batch['B_greyscale'].to(self.device2)

                else:
                    real_A_rgb = batch['A_rgb'].to(self.device1)
                    real_B_rgb = batch['B_rgb'].to(self.device1)
                    real_A_greyscale = batch['A_greyscale'].to(self.device1)
                    real_B_greyscale = batch['B_greyscale'].to(self.device1)

                # Train generator
                gen_error = self.train_generator(real_A_rgb, real_B_rgb, real_A_greyscale, real_B_greyscale)
                self.loss_list.append(gen_error.detach().item())

                # self.plot_metrics(epoch, i)

                self.give_training_eta(start, i, epoch)
                # self.gather_losses(i)

                save_dict = self.get_results(i, epoch, real_A_rgb, real_B_rgb)
                self.save_images(save_dict, self.args, epoch, i)

            self.save_training(epoch)

        self.final_save()

    def train_generator(self, real_A_rgb, real_B_rgb, real_A_greyscale, real_B_greyscale):

        # Set generator gradients to zero
        self.gen_A2B_optimizer.zero_grad()

        if self.A2B:
            self.fake_B = self.gen_A2B(real_A_rgb)
            gen_A2B_loss = self.loss(self.fake_B, real_B_rgb)

        else:
            self.fake_B = self.gen_A2B(real_B_rgb)
            gen_A2B_loss = self.loss(self.fake_B, real_A_rgb, real_A_greyscale)

        # gen_A2B_loss = self.loss(self.fake_B, real_B)
        # print("loss:", gen_A2B_loss)

        # Calculate gradients for both generators
        gen_A2B_loss.backward()

        # Update weights of generator
        self.gen_A2B_optimizer.step()

        return gen_A2B_loss.cpu()

    def initialize_nets(self):
        if self.args.multi_gpu:
            # better to put the model on non-primary gpu
            self.gen_A2B = ColorNet(self.output_dim).to(self.device2)
        else:
            self.gen_A2B = ColorNet(self.output_dim).to(self.device1)

    def load_weights(self):
        if self.args.gen_A2B_dict != '':
            #try:
            self.gen_A2B.load_state_dict(torch.load(self.args.gen_A2B_dict), strict=False)
            #except FileNotFoundError:
                #print("Something went wrong with loading weights!")
                #exit(1)

    def define_optimizers(self):
        self.gen_A2B_optimizer = torch.optim.Adam(self.gen_A2B.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

    def define_schedulers(self):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.args.epochs) / float(self.args.decay_epochs + 1)

        self.gen_A2B_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_A2B_optimizer, lr_lambda=lambda_rule)
        # self.gen_A2B_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.gen_A2B_optimizer, gamma=0.68383)

    def get_results(self, i, epoch, real_A1, real_B1):
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
                    "real_A": real_A1,
                    "real_B": real_B1,
                    "fake_B": self.fake_B
                }

        return tensor_dict

    def save_training(self, epoch):
        """
        Update schedulers, save state_dict of networks, and plot loss metrics.
        """
        # update learning rates
        self.gen_A2B_scheduler.step()

        # checkpoints
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_epoch_{epoch}.pth")

    def plot_metrics(self, epoch, i):
        if self.args.metrics and i % 250 == 0:
            try:
                # plot loss versus epochs
                plt.figure(figsize=[8, 6])
                plt.plot(self.loss_list[(18601 * epoch):], 'r', linewidth=1)
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Loss', fontsize=16)
                plt.savefig(f"figures/epoch{epoch}.png")

            except Exception as e:
                print(e)
                print("Plotting failed...")

    def final_save(self):
        """
         Save final models.
        """
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_final.pth")

    def test(self):
        """
        Test inputs to trained model.
        """
        print("Making directories...")
        self.make_test_directories()

        print("Initializing generators...")
        self.initialize_nets()

        print("Loading weights...")
        self.load_weights()

        print("Setting model mode...")
        self.gen_A2B.eval()

        for i, batch in enumerate(self.loader):
            # get batch data
            if self.args.multi_gpu:
                input_panel_A = batch['A'].to(self.device2)
            else:
                input_panel_A = batch['A'].to(self.device1)

            output_panel_A = 0.5 * (self.gen_A2B(input_panel_A).data + 1.0)

            vutils.save_image(output_panel_A.detach(), f"{self.args.output_path}/A_{i}.png")
