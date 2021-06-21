import argparse
import mode
import os
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--test_path', type=str, default=str(os.getcwd()) + '\\test')
parser.add_argument('--train_path', type=str, default=str(os.getcwd()) + '\\train')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
parser.add_argument('--decay_epochs', type=int, default=100, help='Number of epochs which the learning rate '
                                                                  'decays linearly')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gen_A2B', type=str, default=str(os.getcwd()) + '\\models\\gen_A2B_final.pth',
                    help='Path to the A to B generator')
parser.add_argument('--gen_B2A', type=str, default=str(os.getcwd()) + '\\models\\gen_B2A_final.pth',
                    help='Path to the B to A generator')
parser.add_argument('--dis_A', type=str, default=str(os.getcwd()) + '\\models\\dis_A_final.pth',
                    help='Path to the A discriminator')
parser.add_argument('--dis_B', type=str, default=str(os.getcwd()) + '\\models\\dis_B_final.pth',
                    help='Path to the B discriminator')
parser.add_argument('--save_path', type=str, default=str(os.getcwd()) + '\\models')
parser.add_argument('--image_size', type=int, default=512, help='Size of image resize in training')
parser.add_argument('--output_path', type=str, default=str(os.getcwd()) + '\\results')
parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle data in data loader after each epoch')
parser.add_argument('--metrics', type=bool, default=False, help='Output loss versus number of epochs plot (per epoch)')

args = parser.parse_args()

if __name__ == "__main__":

    if args.mode == 'train':
        mode.train(args)

    elif args.mode == 'test':
        mode.test(args)

    else:
        print("Please enter a valid mode: \'train\' or \'test\'")
