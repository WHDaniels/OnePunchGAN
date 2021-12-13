"""
Not relevant for final approach...
"""

from main.nets import ColorNet, Critic, Generator, Discriminator
from utilities.utils import weights_init, init_weights
from trainers.BaseTrainer import BaseTrainer
from torchvision import utils as vutils
from matplotlib import pyplot as plt
from time import perf_counter
from torch import autograd
import torch
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class WGANGPTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.channels = 1

        self.gen, self.critic = None, None
        self.gen_optimizer, self.gen_scheduler = None, None
        self.critic_optimizer, self.critic_scheduler = None, None

        self.fake_output = None

    def train(self):

        self.initialize_training()

        print("Starting training...")
        for epoch in range(0, self.args.epochs + self.args.decay_epochs):
            self.total_loss_list = list()

            for i, batch in enumerate(self.loader):
                print(f"Epoch {epoch} | Iter {i}")

                start = perf_counter()

                # get batch data (multi_gpu assumed)
                first_B = torch.unsqueeze(batch['B_rgb'][0], 0).to(self.device1)
                rest_A_grey = [torch.unsqueeze(image, 0) for image in batch['A_greyscale'][1:]]
                rest_B_rgb = [torch.unsqueeze(image, 0) for image in batch['B_rgb'][1:]]

                rest = zip(rest_A_grey, rest_B_rgb)

                # Train generator
                # Please re-enable generator saving when training generator
                gen_error = self.train_generator(first_B)

                # Train critic
                total_critic_loss = self.train_critic(rest)

                # if training both
                self.loss_list.append((gen_error.detach().item(), total_critic_loss / self.args.critic_iters))

                # if training critic only
                # self.loss_list.append((0, total_critic_loss / self.args.critic_iters))

                self.plot_metrics(epoch, i)

                real_input, real_output = list(zip(rest_A_grey, rest_B_rgb))[-1]
                save_dict = self.get_results(i, epoch, real_input, real_output)
                self.save_images(save_dict, self.args, epoch, i)
                self.give_training_eta(start, i, epoch)

            self.save_training(epoch)

        self.final_save()

    def train_generator(self, first_B):
        for param in self.critic.parameters():
            param.requires_grad = False

        self.gen.zero_grad()
        self.fake_output = self.gen(first_B)
        critic_generated = self.critic(self.fake_output)
        gen_cost = - critic_generated.mean()
        print('gen_cost', gen_cost)
        gen_cost.backward()
        self.gen_optimizer.step()

        return gen_cost

    def calculate_gradient_penalty(self, real_output):
        alpha = torch.rand(self.args.batch_size, 1)
        alpha = alpha.expand(self.args.batch_size,
                             int(real_output.nelement() / self.args.batch_size)).contiguous()
        alpha = alpha.view(self.args.batch_size, self.channels, self.args.image_size, self.args.image_size)
        alpha = alpha.to(self.device1)

        interpolated = alpha * real_output + ((1 - alpha) * self.fake_output)
        interpolated = interpolated.to(self.device1)
        interpolated.requires_grad_(True)

        interpolated = autograd.Variable(interpolated, requires_grad=True)
        critic_interpolated = self.critic(interpolated)

        gradients = autograd.grad(outputs=critic_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(critic_interpolated.size()).to(self.device1),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(real_output.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1.0) ** 2).mean() * 10

        return gradient_penalty

    def train_critic(self, rest):
        for param in self.critic.parameters():
            param.requires_grad = True

        total_critic_loss = 0
        for A, B in rest:
            self.critic.zero_grad()
            real_output = A.to(self.device1)  # greyscale real output
            real_input = B.to(self.device1)  # colored real input

            self.critic.zero_grad()

            with torch.no_grad():
                gen_input = real_input  # totally freeze G, training D
            self.fake_output = self.gen(gen_input).detach()

            # if training B2A:
            # real_output here should be a 1-channel greyscale image (take 1-channel version of image from dataloader)
            real_pred = self.critic(real_output)

            # and fake_B should be a 1-channel greyscale image (set ColorNet(dim=1), not hard)
            fake_pred = self.critic(self.fake_output)

            # calculate gradient penalty
            # here want 1-channel version of real_input as well, to match fake_B
            gradient_penalty = self.calculate_gradient_penalty(real_output)

            # final critic cost
            critic_cost = fake_pred.mean() - real_pred.mean() + gradient_penalty

            critic_cost.backward()

            print("total", critic_cost)
            print("fake", fake_pred.mean())
            print("real_pred", real_pred.mean())
            print("both", fake_pred.mean() - real_pred.mean())
            print("gp", gradient_penalty)
            print("\n")

            self.critic_optimizer.step()
            total_critic_loss += critic_cost.cpu().detach().item()

        return total_critic_loss

    def initialize_nets(self):
        if self.args.multi_gpu:
            self.gen = Generator(3, 1).to(self.device1)
            # self.gen = ColorNet(self.channels).to(self.device1)
            # self.critic = Discriminator(self.channels).to(self.device1)
            self.critic = Critic(output_dim=self.channels).to(self.device1)

            self.gen.apply(init_weights)
            self.critic.apply(init_weights)
            # self.dis_B.apply(weights_init)

        else:
            raise NotImplementedError('Single GPU for WGAN-GP not implemented!')

    def load_weights(self):
        if self.args.gen_A2B_dict != '':
            self.gen.load_state_dict(torch.load(self.args.gen_A2B_dict))

        if self.args.dis_A_dict != '':
            self.critic.load_state_dict(torch.load(self.args.dis_A_dict))

    def define_optimizers(self):
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.args.lr / 4, betas=(0.5, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

    def define_schedulers(self):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.args.epochs) / float(self.args.decay_epochs + 1)

        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_optimizer, lr_lambda=lambda_rule)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda_rule)

    def get_results(self, i, epoch, real_A, real_B):
        """
        Show results after a multiple of iterations.
        """
        tensor_dict = {}

        if i % 50 == 0:
            # show results on test per epoch
            try:
                os.mkdir(os.getcwd() + f'\\results\\epoch_{epoch}')

            except FileExistsError:
                print("Directory exists already...")

            finally:
                # Save image
                tensor_dict = {
                    "real_A": real_A,
                    "real_B": real_B,
                    "fake_B": self.fake_output
                }

        return tensor_dict

    def save_training(self, epoch):
        """
        Update schedulers, save state_dict of networks, and plot loss metrics.
        """
        # update learning rates
        self.gen_scheduler.step()
        self.critic_scheduler.step()

        # checkpoints
        torch.save(self.gen.state_dict(), f"{self.args.save_path}/gan_gen_epoch_{epoch}.pth")
        torch.save(self.critic.state_dict(), f"{self.args.save_path}/gan_critic_epoch_{epoch}.pth")

    def plot_metrics(self, epoch, i):
        if self.args.metrics and i % 50 == 0:
            loss_gen = [loss[0] for loss in self.loss_list]
            loss_critic = [loss[1] for loss in self.loss_list]
            try:
                # plot loss versus epochs
                plt.figure(figsize=[8, 6])
                plt.ylim(-20, 20)
                plt.plot(loss_gen, 'r', linewidth=1)
                plt.plot(loss_critic, 'b', linewidth=1)
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Loss', fontsize=16)
                plt.savefig(f"figures/epoch{epoch}.png")

            except Exception as e:
                print(e)
                print("Plotting failed...")

    def final_save(self):
        torch.save(self.gen.state_dict(), f"{self.args.save_path}/gan_gen_final.pth")
        torch.save(self.critic.state_dict(), f"{self.args.save_path}/gan_critic_final.pth")

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
        self.gen.eval()

        for i, batch in enumerate(self.loader):
            # get batch data
            if self.args.multi_gpu:
                # device 2 below
                input_panel_A = batch['A'].to(self.device1)
            else:
                input_panel_A = batch['A'].to(self.device1)

            output_panel_A = 0.5 * (self.gen(input_panel_A).data + 1.0)

            vutils.save_image(output_panel_A.detach(), f"{self.args.output_path}/A_{i}.png")
