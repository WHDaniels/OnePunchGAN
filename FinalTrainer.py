from utils import save_images, init_weights, UnNormalize
from nets import ColorNet, Discriminator, Generator
from torchvision import transforms as ts
from matplotlib import pyplot as plt
from BaseTrainer import BaseTrainer
import torchvision.utils as vutils
from image_pool import ImagePool
from time import perf_counter
from torch import nn
import torch
import os


class FinalTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.adversarial_loss = torch.nn.MSELoss().to(self.device2)
        self.cycle_consistency_loss = torch.nn.L1Loss().to(self.device2)

        self.gen_A2B_optimizer, self.gen_A2B_scheduler = None, None
        self.dis_optimizer, self.dis_scheduler = None, None

        self.gen_A2B, self.gen_B2A, self.dis = None, None, None
        self.identity_B = None

        self.fake_A, self.fake_B = None, None
        self.real_A, self.real_A_norm = None, None

        self.fake_B_pool = ImagePool(50)

    def train(self):

        self.initialize_training()

        print("Starting training...")
        for epoch in range(0, self.args.epochs + self.args.decay_epochs):
            self.total_loss_list = list()

            for i, batch in enumerate(self.loader):
                print(f"Epoch {epoch} | Iter {i}")

                start = perf_counter()

                # just get real B
                real_B_reg_norm = batch['real_B_reg_norm'].to(self.device2)
                real_B_reg_norm2 = batch['real_B_reg_norm'].to(self.device2)

                # Get "real" black and white input
                with torch.no_grad():
                    self.real_A_norm = self.gen_B2A(real_B_reg_norm).detach()

                # normalize input image for inputting to pretrained backbone network
                self.real_A = self.normalize_to_pretrained()

                # Train generator
                gen_A2B_loss = self.train_generators(real_B_reg_norm2, i, epoch)
                # self.gen_A2B_loss_list.append(gen_A2B_loss.cpu().detach().numpy())

                # Train discriminator
                dis_B_loss = self.train_discriminator_B(real_B_reg_norm2)
                # self.dis_B_loss_list.append(dis_B_loss.cpu().detach().numpy())

                self.give_training_eta(start, i, epoch)
                save_dict = self.get_results(i, epoch, self.real_A, real_B_reg_norm2)
                save_images(save_dict, self.args, epoch, i)

            self.save_training(epoch)

        self.final_save()

    def train_generators(self, real_B2, i, epochs):

        # Freeze discriminators
        self.dis.requires_grad_(False)

        # Set generator gradient to zero
        self.gen_A2B_optimizer.zero_grad(set_to_none=True)

        self.fake_B = self.gen_A2B(self.real_A.detach())

        # Getting adversarial loss
        gen_A2B_loss = self.adversarial_loss(self.dis(self.fake_B), self.target_real2)

        # B input to cycle loss here is real B (colored image)
        cycle_BAB_loss = self.cycle_consistency_loss(self.fake_B, real_B2) * 10

        # Combine loss
        gen_error = gen_A2B_loss + cycle_BAB_loss

        # Calculate gradients for generator
        gen_error.backward()

        # Update weights of generator
        self.gen_A2B_optimizer.step()

        if i % 50 == 0:
            total_epochs = self.args.epochs + self.args.decay_epochs
            print(f"\nDataset passes: {epochs / total_epochs}")
            if epochs > self.args.decay_epochs:
                print(
                    f"Discriminator learning rate: {((self.args.decay_epochs - (epochs - self.args.decay_epochs)) / self.args.decay_epochs) * self.args.lr}")
                print(
                    f"Generator learning rate: {((self.args.decay_epochs - (epochs - self.args.decay_epochs)) / self.args.decay_epochs) * self.args.lr / 4}\n")
            print("cycle_BAB_loss", cycle_BAB_loss.detach().cpu())
            print("gen_A2B_loss", gen_A2B_loss.detach().cpu(), "\n")

        return gen_error

    def train_discriminator_B(self, real):
        # Unfreeze discriminators
        self.dis.requires_grad_(True)

        # Set discriminator gradients to zero
        self.dis_optimizer.zero_grad(set_to_none=True)

        # Loss for discriminator A
        new_fake = self.fake_B_pool.query(self.fake_B)

        # Getting predictions
        real_prediction = self.dis(real)
        fake_prediction = self.dis(new_fake.detach())

        # Getting discriminator loss
        dis_real_loss = self.adversarial_loss(real_prediction, self.target_real2)
        dis_fake_loss = self.adversarial_loss(fake_prediction, self.target_fake2)

        # Calculating discriminator error
        dis_B_loss = (dis_real_loss + dis_fake_loss) * 0.5

        # Gradient update
        dis_B_loss.backward()

        # update discriminator weights
        self.dis_optimizer.step()

        return dis_B_loss

    def initialize_nets(self):
        if self.args.multi_gpu:
            self.gen_A2B = ColorNet(self.channels).to(self.device2)

            # custom ColorNet init
            for m in list(self.gen_A2B.children())[5:]:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif isinstance(m, nn.InstanceNorm2d):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)

            rest = False
            for x, param in self.gen_A2B.named_parameters():
                if rest:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                if 'block5.2.bn3.bias' in x:
                    rest = True

            self.gen_B2A = Generator(self.channels, 1).eval().to(self.device2)
            self.dis = Discriminator(3).to(self.device2)
            self.dis.apply(init_weights)

        else:
            self.gen_A2B = ColorNet().to(self.device1)
            self.gen_B2A = ColorNet().to(self.device1)
            self.dis = Discriminator(3).to(self.device1)
            self.dis.apply(init_weights)

    def load_weights(self):
        # if generator or discriminator path specified, load them from directory
        if self.args.gen_A2B_dict != '':
            self.gen_A2B.load_state_dict(torch.load(self.args.gen_A2B_dict))

        if self.args.gen_B2A_dict != '':
            self.gen_B2A.load_state_dict(torch.load(self.args.gen_B2A_dict))

        if self.args.dis_B_dict != '':
            self.dis.load_state_dict(torch.load(self.args.dis_B_dict))

    def define_optimizers(self):
        self.gen_A2B_optimizer = torch.optim.Adam(self.gen_A2B.parameters(), lr=self.args.lr / 4, betas=(0.5, 0.999))
        self.dis_optimizer = torch.optim.Adam(self.dis.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

    def define_schedulers(self):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.args.epochs) / float(self.args.decay_epochs + 1)

        self.gen_A2B_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_A2B_optimizer, lr_lambda=lambda_rule)
        self.dis_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_optimizer, lr_lambda=lambda_rule)

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

    def plot_metrics(self, epoch, i, per):
        self.avg_gen_loss.append(sum([loss for loss in self.gen_A2B_loss_list[i - per:]])
                                 / len(self.gen_A2B_loss_list[i - per:]) / self.args.critic_iters)
        self.avg_dis_loss.append(sum([loss for loss in self.dis_B_loss_list[i - per:]])
                                 / len(self.dis_B_loss_list[i - per:]))

        try:
            # plot loss versus epochs
            plt.figure(figsize=[8, 6])
            plt.ylim(-1, 3.5)
            plt.plot(self.avg_gen_loss, 'r', linewidth=1)
            plt.plot(self.avg_dis_loss, 'b', linewidth=1)
            plt.xlabel(f'Iterations per thousand', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.savefig(f"figures/epoch{epoch}_iter{i}.png")

        except Exception as e:
            print(e)
            print("Plotting failed...")

    def save_training(self, epoch):

        # update learning rates
        self.gen_A2B_scheduler.step()
        self.dis_scheduler.step()

        # checkpoints
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_epoch_{epoch}.pth")
        torch.save(self.dis.state_dict(), f"{self.args.save_path}/dis_B_epoch_{epoch}.pth")

    def final_save(self):
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_final.pth")
        torch.save(self.dis.state_dict(), f"{self.args.save_path}/dis_B_final.pth")

    def normalize_to_pretrained(self, opt_image=None):
        if opt_image is None:
            image_tensor = self.real_A_norm.data
        else:
            image_tensor = opt_image
        image_tensor = torch.tile(image_tensor, (3, 1, 1))
        gen_input = UnNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor)
        gen_image = ts.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979])(gen_input)
        return gen_image

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
            if self.args.multi_gpu:
                input_panel_A = batch['real_B'].to(self.device2)
            else:
                input_panel_A = batch['real_B'].to(self.device2)

            input_panel = self.gen_B2A(input_panel_A).squeeze(0)
            input_panel = 0.5 * (input_panel + 1.0)
            input_panel = ts.ToPILImage()(input_panel).convert('RGB')
            input_panel = ts.ToTensor()(input_panel).unsqueeze(0)
            input_panel = ts.Normalize([0.7137, 0.6628, 0.6519], [0.2970, 0.3017, 0.2979])(input_panel).to(self.device2)

            output_panel_A = self.gen_A2B(input_panel).data
            image = 0.5 * (output_panel_A + 1.0)
            vutils.save_image(image.detach(), f"{self.args.output_path}/A_{i}.png")
            print("Complete!")
