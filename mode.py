import gc

from utils import *
from dataset import *
import itertools
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from image_pool import ImagePool
from time import perf_counter
import matplotlib.pyplot as plt
import os

# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    print("Making directories...")
    make_train_directories(args)

    print("Getting paths...")
    train_A_path, train_B_path = os.path.join(args.train_path, 'train_A'), os.path.join(args.train_path, 'train_B')

    print("Getting dataset and dataloader...")
    train_dataset = PanelDataset(train_A_path, train_B_path, args.image_size, args.mode)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle)

    print("Initializing generators and discriminators...")
    gen_A2B, gen_B2A, dis_A, dis_B = initialize_nets(args, device)

    print("Defining losses...")
    adversarial_loss = torch.nn.MSELoss().to(device)
    cycle_consistency_loss = torch.nn.L1Loss().to(device)
    identity_loss = torch.nn.L1Loss().to(device)

    print("Defining optimizers...")
    gen_optimizer = torch.optim.Adam(itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()), lr=args.lr,
                                     betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(itertools.chain(dis_A.parameters(), dis_B.parameters()), lr=args.lr,
                                     betas=(0.5, 0.999))

    print("Creating image pools...")
    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - args.epochs) / float(args.decay_epochs + 1)

    print("Defining schedulers...")
    gen_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=lambda_rule)
    dis_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(dis_optimizer, lr_lambda=lambda_rule)

    print("Defining loss gathering metrics...")
    loss_list = list()

    print("Starting training...")
    for epoch in range(0, args.epochs):

        print(f"Epoch {epoch}")
        for i, batch in enumerate(train_loader):
            start = perf_counter()

            # get batch data
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Freeze discriminators
            dis_A.requires_grad_(False)
            dis_B.requires_grad_(False)

            # Generator Training
            # Set both generators gradients to zero
            gen_optimizer.zero_grad()

            # Getting identity loss
            lambda_A, lambda_B, lambda_identity = 10.0, 10.0, 0.5

            identity_A = gen_A2B(real_B)  # G_A should be identity if real_B is fed
            identity_A_loss = identity_loss(identity_A, real_B) * lambda_B * lambda_identity

            identity_B = gen_B2A(real_A)  # G_B should be identity if real_A is fed
            identity_B_loss = identity_loss(identity_B, real_A) * lambda_A * lambda_identity

            # Free up memory by deleting unneeded tensors
            # del identity_A, identity_B
            # gc.collect()
            # torch.cuda.empty_cache()

            # Getting adversarial loss
            target_real = torch.full((args.batch_size, 1, 30, 30), 1, device=device, dtype=torch.float32)
            target_fake = torch.full((args.batch_size, 1, 30, 30), 0, device=device, dtype=torch.float32)

            # get fakes and recreation of real from fake
            fake_B = gen_A2B(real_A)
            rec_A = gen_B2A(fake_B)
            fake_A = gen_B2A(real_B)
            rec_B = gen_A2B(fake_A)

            gen_A2B_loss = adversarial_loss(dis_A(fake_B), target_real)  # GAN loss D_A(G_A(A))
            gen_B2A_loss = adversarial_loss(dis_B(fake_A), target_real)  # GAN loss D_B(G_B(B))

            # Getting cycle consistency loss
            cycle_A_loss = cycle_consistency_loss(rec_A, real_A) * lambda_A
            cycle_B_loss = cycle_consistency_loss(rec_B, real_B) * lambda_B

            # Free up memory by deleting unneeded tensors
            # del rec_A, rec_B
            # torch.cuda.empty_cache()

            # Combine loss
            gen_error = identity_A_loss + identity_B_loss + gen_A2B_loss + gen_B2A_loss + cycle_A_loss + cycle_B_loss

            # Calculate gradients for both generators
            gen_error.backward()

            # Update weights of both generators
            gen_optimizer.step()

            # Discriminator training
            # Unfreeze discriminators
            dis_A.requires_grad_(True)
            dis_B.requires_grad_(True)

            # Set discriminator gradients to zero
            dis_optimizer.zero_grad()

            # Loss for discriminator A
            fake = fake_B_pool.query(fake_B)

            # Getting predictions
            real_prediction = dis_A(real_B)
            fake_prediction = dis_A(fake.detach())

            # Getting discriminator A's loss
            dis_A_real_loss = adversarial_loss(real_prediction, target_real)
            dis_A_fake_loss = adversarial_loss(fake_prediction, target_fake)

            # Calculating discriminator A's error
            dis_A_error = (dis_A_real_loss * dis_A_fake_loss) * 0.5

            # Gradient update
            dis_A_error.backward()

            # Loss for discriminator B
            fake = fake_A_pool.query(fake_A)

            # Getting predictions
            real_prediction = dis_B(real_A)
            fake_prediction = dis_B(fake.detach())

            # Getting discriminator B's loss
            dis_B_real_loss = adversarial_loss(real_prediction, target_real)
            dis_B_fake_loss = adversarial_loss(fake_prediction, target_fake)

            # Calculating discriminator B's error
            dis_B_error = (dis_B_real_loss * dis_B_fake_loss) * 0.5

            # Gradient update
            dis_B_error.backward()

            # update discriminator weights
            dis_optimizer.step()

            # Give a training ETA
            end = perf_counter()

            if i % 25 == 0:
                perBatch = end - start
                print('-' * 50)
                print(f"\nTime per batch: {perBatch} seconds")

                epochFinish = perBatch * len(train_dataset)
                print(f"Time until epoch finished: {epochFinish / 3600} hours")

                totalEpochs = args.epochs + args.decay_epochs
                print(f"Time until training is completed: {epochFinish * totalEpochs / 86400} days\n")
                print('-' * 50)

            if i + 1 == len(train_loader):
                loss_list.append(gen_error.cpu().detach().numpy())

        # checkpoints
        torch.save(gen_A2B.state_dict(), f"{args.save_path}/gen_A2B_epoch_{epoch}.pth")
        torch.save(gen_B2A.state_dict(), f"{args.save_path}/gen_B2A_epoch_{epoch}.pth")
        torch.save(dis_A.state_dict(), f"{args.save_path}/dis_A_epoch_{epoch}.pth")
        torch.save(dis_B.state_dict(), f"{args.save_path}/dis_B_epoch_{epoch}.pth")

        # update learning rates
        gen_lr_scheduler.step()
        dis_lr_scheduler.step()

        if args.metrics:
            # plot loss versus epochs
            plt.figure(figsize=[8, 6])
            plt.plot(loss_list, 'r', linewidth=3)
            plt.xlabel('Epochs', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.savefig(f"figures/epoch{epoch}.png")

    # save last check pointing
    torch.save(gen_A2B.state_dict(), f"{args.save_path}/gen_A2B_final.pth")
    torch.save(gen_B2A.state_dict(), f"{args.save_path}/gen_B2A_final.pth")
    torch.save(dis_A.state_dict(), f"{args.save_path}/dis_A_final.pth")
    torch.save(dis_B.state_dict(), f"{args.save_path}/dis_B_final.pth")


def test(args):
    print("Making directories...")
    make_test_directories(args)

    print("Getting dataset and dataloader...")
    test_dataset = PanelDataset(data_path_A=args.test_path, data_path_B=args.test_path, image_size=args.image_size,
                                mode=args.mode)
    test_loader = DataLoader(test_dataset, args.batch_size)

    print("Initializing generators...")
    gen_A2B, gen_B2A, _, _ = initialize_nets(args, device)

    print("Loading generators weights...")
    gen_A2B.load_state_dict(torch.load(args.gen_A2B))
    gen_B2A.load_state_dict(torch.load(args.gen_B2A))

    print("Setting model mode...")
    gen_A2B.eval()
    gen_B2A.eval()

    for i, batch in enumerate(test_loader):
        # get batch data
        input_panel_A = batch['A'].to(device)
        input_panel_B = batch['B'].to(device)

        # Generate images
        output_panel_A = 0.5 * (gen_A2B(input_panel_A).data + 1.0)
        output_panel_B = 0.5 * (gen_B2A(input_panel_B).data + 1.0)
        # output_panel_A = gen_A2B(input_panel_A)
        # output_panel_B = gen_B2A(input_panel_B)

        # Save image
        vutils.save_image(output_panel_A.detach(), f"{args.output_path}/A_{i}.png")
        vutils.save_image(output_panel_B.detach(), f"{args.output_path}/B_{i}.png")
