import argparse

parser = argparse.ArgumentParser()

# path to dataset to train/test on
# epochs to train on
# when to start linearly decaying the learning rate to 0
# batch size
# learning rate
# path to generators and discriminators
# image size
# folder to output images
# seed for training

parser.add_argument('--test_path', type=str, default='./test')
parser.add_argument('--train_path', type=str, default='./train')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
parser.add_argument('--decay_epochs', type=int, default=100, help='Number of epochs which the learning rate '
                                                                  'decays linearly')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gen_A2B', type=str, default='./models/genA2B_final', help='Path to the A to B generator')
parser.add_argument('--gen_B2A', type=str, default='./models/genB2A_final', help='Path to the B to A generator')
parser.add_argument('--dis_A', type=str, default='./models/disA_final', help='Path to the A discriminator')
parser.add_argument('--dis_B', type=str, default='./models/disB_final', help='Path to the B discriminator')
parser.add_argument('--image_size', type=int, default=512, help='Size of image resize in training')
parser.add_argument('--output_path', type=str, default='./output')
parser.add_argument('--seed', type=int, help='Seed for initializing training')
