import os
import torch
from time import perf_counter
from image_pool import ImagePool
import torchvision.utils as vutils
from BaseTrainer import BaseTrainer
from matplotlib import pyplot as plt
from nets import ColorNet, Discriminator, Generator
from utils import save_images, weights_init, init_weights


class CycleTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.adversarial_loss1 = torch.nn.MSELoss().to(self.device1)
        self.cycle_consistency_loss1 = torch.nn.L1Loss().to(self.device1)
        self.identity_loss1 = torch.nn.L1Loss().to(self.device1)

        if self.args.multi_gpu:
            self.adversarial_loss2 = torch.nn.MSELoss().to(self.device2)
            self.cycle_consistency_loss2 = torch.nn.L1Loss().to(self.device2)
            self.identity_loss2 = torch.nn.L1Loss().to(self.device2)

        self.gen_A2B_optimizer, self.gen_B2A_optimizer = None, None
        self.dis_A_optimizer, self.dis_B_optimizer = None, None

        self.gen_A2B_scheduler, self.gen_B2A_scheduler = None, None
        self.dis_A_scheduler, self.dis_B_scheduler = None, None

        self.gen_A2B, self.gen_B2A = None, None
        self.dis_A, self.dis_B = None, None

        self.identity_A, self.identity_B = None, None

        self.fake_A, self.fake_B = None, None
        self.rec_A, self.rec_B = None, None

        self.fake_A_pool = ImagePool(50)
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

                # get batch data
                real_A1 = batch['A_rgb'].to(self.device1)
                real_A2 = batch['A_rgb'].to(self.device2)
                real_A1_grey = batch['A_greyscale'].to(self.device1)
                real_A2_grey = batch['A_greyscale'].to(self.device2)

                real_B1 = batch['B_rgb'].to(self.device1)
                real_B2 = batch['B_rgb'].to(self.device2)
                real_B1_grey = batch['B_greyscale'].to(self.device1)
                real_B2_grey = batch['B_greyscale'].to(self.device2)

                batch_list = [real_A1, real_A2, real_A1_grey, real_A2_grey,
                              real_B1, real_B2, real_B1_grey, real_B2_grey]

                # Freeze discriminators
                self.dis_A.requires_grad_(False)
                self.dis_B.requires_grad_(False)

                # Train both generators
                gen_A2B_loss, gen_B2A_loss = self.train_generators(batch_list)
                gen_A2B_loss = self.train_generators(batch_list)
                self.gen_A2B_loss_list.append(gen_A2B_loss.cpu().detach().numpy())
                self.gen_B2A_loss_list.append(gen_B2A_loss.cpu().detach().numpy())

                # Train both discriminators
                dis_A_loss, dis_B_loss = self.train_discriminator_A(real_A2_grey), self.train_discriminator_B(real_B1)
                self.dis_A_loss_list.append(dis_A_loss.cpu().detach().numpy())
                self.dis_B_loss_list.append(dis_B_loss.cpu().detach().numpy())

                self.give_training_eta(start, i, epoch)
                # self.gather_losses(i)

                save_dict = self.get_results(i, epoch, real_A1, real_B1)
                save_images(save_dict, self.args, epoch, i)

            self.save_training(epoch)

        self.final_save()

    def train_generators(self, batch_list):

        real_A1, real_A2, real_A1_grey, real_A2_grey = batch_list[0:4]
        real_B1, real_B2, real_B1_grey, real_B2_grey = batch_list[4:8]

        # Set both generators gradients to zero
        self.gen_A2B_optimizer.zero_grad()
        self.gen_B2A_optimizer.zero_grad()

        # Getting identity loss
        lambda_A, lambda_B, lambda_identity = 10.0, 10.0, 0.5

        self.identity_A = self.gen_B2A(real_A2)
        identity_A_loss = self.identity_loss1(self.identity_A, real_A2_grey) * lambda_B * lambda_identity

        self.identity_B = self.gen_A2B(real_B1_grey)
        identity_B_loss = self.identity_loss2(self.identity_B, real_B1) * lambda_A * lambda_identity

        # Getting adversarial loss
        self.fake_B = self.gen_A2B(real_A1_grey)
        self.fake_A = self.gen_B2A(real_B2)

        gen_A2B_loss = self.adversarial_loss1(self.dis_B(self.fake_B), self.target_real1)  # GAN loss D_A(G_A(A))
        gen_B2A_loss = self.adversarial_loss2(self.dis_A(self.fake_A), self.target_real2)  # GAN loss D_B(G_B(B))

        self.rec_B = self.gen_A2B(self.fake_A.to(self.device1))
        self.rec_A = self.gen_B2A(self.fake_B.to(self.device2))

        # Getting cycle consistency loss
        cycle_ABA_loss = self.cycle_consistency_loss1(self.rec_A, real_A2_grey) * lambda_A
        cycle_BAB_loss = self.cycle_consistency_loss2(self.rec_B, real_B1) * lambda_B

        # Combine loss
        gen_error = identity_A_loss.cpu() + identity_B_loss.cpu() + gen_A2B_loss.cpu() + \
                    gen_B2A_loss.cpu() + cycle_ABA_loss.cpu() + cycle_BAB_loss.cpu()

        # Calculate gradients for both generators
        gen_error.backward()

        # Update weights of both generators
        self.gen_A2B_optimizer.step()
        self.gen_B2A_optimizer.step()

        # return gen_A2B_loss.cpu(), gen_B2A_loss.cpu()
        return gen_A2B_loss.cpu()

    def train_discriminator_A(self, real):
        # Unfreeze discriminators
        self.dis_A.requires_grad_(True)

        # Set discriminator gradients to zero
        self.dis_A_optimizer.zero_grad()

        # Loss for discriminator A
        new_fake = self.fake_A_pool.query(self.fake_A)

        # Getting predictions
        real_prediction = self.dis_A(real)
        fake_prediction = self.dis_A(new_fake.detach())

        # Getting discriminator loss
        dis_real_loss = self.adversarial_loss2(real_prediction, self.target_real2)
        dis_fake_loss = self.adversarial_loss2(fake_prediction, self.target_fake2)

        # Calculating discriminator error
        dis_A_loss = (dis_real_loss + dis_fake_loss) * 0.5

        # Gradient update
        dis_A_loss.backward()

        # update discriminator weights
        self.dis_A_optimizer.step()

        return dis_A_loss

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
        dis_real_loss = self.adversarial_loss1(real_prediction, self.target_real1)
        dis_fake_loss = self.adversarial_loss1(fake_prediction, self.target_fake1)

        # Calculating discriminator error
        dis_B_loss = (dis_real_loss + dis_fake_loss) * 0.5

        # Gradient update
        dis_B_loss.backward()

        # update discriminator weights
        self.dis_B_optimizer.step()

        return dis_B_loss

    def initialize_nets(self):
        if self.args.multi_gpu:
            self.gen_A2B = ColorNet(3).to(self.device1)

            for param in self.gen_A2B.parameters():
                print(param)
                # Freeze params in backbone
                exit()
                param.requires_grad = False

            self.gen_B2A = Generator(3, 1).to(self.device2)
            self.dis_A = Discriminator(1).to(self.device2)
            self.dis_B = Discriminator(3).to(self.device1)

            # maybe remove later (#weights_init)
            self.gen_A2B.apply(init_weights)
            self.gen_B2A.apply(init_weights)

            self.dis_A.apply(init_weights)
            self.dis_B.apply(init_weights)

        else:
            self.gen_A2B = ColorNet().to(self.device1)
            self.gen_B2A = ColorNet().to(self.device1)
            self.dis_A = Discriminator(1).to(self.device1)
            self.dis_B = Discriminator(3).to(self.device1)
            self.dis_A.apply(init_weights)
            self.dis_B.apply(init_weights)

    def load_weights(self):
        # if generator or discriminator path specified, load them from directory
        if self.args.gen_A2B_dict != '':
            self.gen_A2B.load_state_dict(torch.load(self.args.gen_A2B_dict), strict=False)

        if self.args.gen_B2A_dict != '':
            """
            model = torch.load(self.args.gen_B2A_dict)

            del model['model.1.weight']
            del model['model.1.bias']
            del model['model.25.weight']
            del model['model.25.bias']
            del model['model.28.weight']
            del model['model.28.bias']
            del model['model.32.weight']
            del model['model.32.bias']

            self.gen_B2A.load_state_dict(model, strict=False)

            # freeze encoder
            for name, param in self.gen_B2A.named_parameters():
                if '25' in name:
                    break
                param.requires_grad = False
            """
            self.gen_B2A.load_state_dict(torch.load(self.args.gen_B2A_dict))

        if self.args.dis_A_dict != '':
            self.dis_A.load_state_dict(torch.load(self.args.dis_A_dict))

        if self.args.dis_B_dict != '':
            self.dis_B.load_state_dict(torch.load(self.args.dis_B_dict))

    def define_optimizers(self):
        self.gen_A2B_optimizer = torch.optim.Adam(self.gen_A2B.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.gen_B2A_optimizer = torch.optim.Adam(self.gen_B2A.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.dis_A_optimizer = torch.optim.Adam(self.dis_A.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.dis_B_optimizer = torch.optim.Adam(self.dis_B.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

    def define_schedulers(self):
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.args.epochs) / float(self.args.decay_epochs + 1)

        self.gen_A2B_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_A2B_optimizer, lr_lambda=lambda_rule)
        self.gen_B2A_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_B2A_optimizer, lr_lambda=lambda_rule)
        self.dis_A_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_A_optimizer, lr_lambda=lambda_rule)
        self.dis_B_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_B_optimizer, lr_lambda=lambda_rule)

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
                    "fake_B": self.fake_B,
                    "rec_A": self.rec_A,
                    "idt_B": self.identity_B,
                    "real_B": real_B1,
                    "fake_A": self.fake_A,
                    "rec_B": self.rec_B,
                    "idt_A": self.identity_A
                }

        return tensor_dict

    def save_training(self, epoch):

        # update learning rates
        self.gen_A2B_scheduler.step()
        self.gen_B2A_scheduler.step()
        self.dis_A_scheduler.step()
        self.dis_B_scheduler.step()

        # checkpoints
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_epoch_{epoch}.pth")
        torch.save(self.gen_B2A.state_dict(), f"{self.args.save_path}/gen_B2A_epoch_{epoch}.pth")
        torch.save(self.dis_A.state_dict(), f"{self.args.save_path}/dis_A_epoch_{epoch}.pth")
        torch.save(self.dis_B.state_dict(), f"{self.args.save_path}/dis_B_epoch_{epoch}.pth")

        if self.args.metrics:
            # plot loss versus epochs
            plt.figure(figsize=[8, 6])
            plt.plot(self.gen_A2B_loss_list, 'r', linewidth=2, label='gen_A2B')
            plt.plot(self.gen_B2A_loss_list, 'm', linewidth=2, label='gen_B2A')
            plt.plot(self.dis_A_loss_list, 'b', linewidth=2, label='dis_A')
            plt.plot(self.dis_B_loss_list, 'c', linewidth=2, label='dis_B')
            plt.xlabel('Epochs', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.legend()
            plt.savefig(f"figures/epoch{epoch}.png")

    def final_save(self):
        torch.save(self.gen_A2B.state_dict(), f"{self.args.save_path}/gen_A2B_final.pth")
        torch.save(self.gen_B2A.state_dict(), f"{self.args.save_path}/gen_B2A_final.pth")
        torch.save(self.dis_A.state_dict(), f"{self.args.save_path}/dis_A_final.pth")
        torch.save(self.dis_B.state_dict(), f"{self.args.save_path}/dis_B_final.pth")

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
