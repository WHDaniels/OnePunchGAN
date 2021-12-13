from trainers.BaseTrainer import BaseTrainer
from torchvision import utils as vutils
from matplotlib import pyplot as plt
from main.nets import Critic
from torch import autograd
import torch
import os


class CriticTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.critic = None
        self.critic_optimizer = None
        self.critic_scheduler = None

    def train(self):
        for epoch in range(0, self.args.epochs + self.args.decay_epochs):
            self.total_loss_list = list()

            for i, batch in enumerate(self.loader):
                print(f"Epoch {epoch} | Iter {i}")

            torch.save(self.critic.state_dict(), f"{self.args.save_path}/gan_critic_epoch_{epoch}.pth")

    def train_critic(self, real_input, fake_input):
        self.critic.requires_grad_(True)
        self.critic.zero_grad()

        real_pred = self.critic(real_input)
        fake_pred = self.critic(fake_input)

        gradient_penalty = self.calc_gradient_penalty(real_input, fake_input)

        # final critic cost
        critic_cost = fake_pred - real_pred + gradient_penalty

        # print(fake_pred)
        # print(real_pred)
        # print(gradient_penalty)
        # print(critic_cost)

        critic_cost.backward()
        self.critic_optimizer.step()

        # get wasserstien distance
        # fake_pred - real_pred

        return critic_cost.cpu().detach().item()

    def calc_gradient_penalty(self, real_data, fake_data, gp_lambda=10):
        alpha = torch.rand(self.args.batch_size, 1)
        alpha = alpha.expand(self.args.batch_size, int(real_data.nelement() / self.args.batch_size)).contiguous()
        # out dim here
        alpha = alpha.view(self.args.batch_size, self.channels, self.args.image_size, self.args.image_size)
        alpha = alpha.to(self.device1)

        # out dim here
        fake_data = fake_data.view(self.args.batch_size, self.channels, self.args.image_size, self.args.image_size)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device1)
        interpolates.requires_grad_(True)

        critic_interpolates = self.critic(interpolates)

        gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(critic_interpolates.size()).to(self.device1),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

    def initialize_nets(self):
        if self.args.multi_gpu:
            self.critic = Critic(self.args.image_size, self.output_dim).to(self.device1)
        else:
            raise NotImplementedError('Single GPU for WGAN-GP not implemented!')

    def load_weights(self):
        if self.args.dis_A_dict != '':
            try:
                self.critic.load_state_dict(torch.load(self.args.dis_A_dict))
            except FileNotFoundError:
                print("Could not find critic state dictionary...")
                pass

    def define_optimizers(self):
        try:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        except AttributeError:
            raise AttributeError("Please initialize critic optimizer...")

    def define_schedulers(self):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.args.epochs) / float(self.args.decay_epochs + 1)

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

    def plot_metrics(self, epoch, i):
        if self.args.metrics and i % 250 == 0:
            try:
                # plot loss versus epochs
                plt.figure(figsize=[8, 6])
                plt.plot(self.loss_list, 'r', linewidth=1)
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Loss', fontsize=16)
                plt.savefig(f"figures/epoch{epoch}.png")

            except Exception as e:
                print(e)
                print("Plotting failed...")

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
                input_panel_A = batch['A'].to(self.device2)
            else:
                input_panel_A = batch['A'].to(self.device1)

            output_panel_A = 0.5 * (self.gen(input_panel_A).data + 1.0)

            vutils.save_image(output_panel_A.detach(), f"{self.args.output_path}/A_{i}.png")
