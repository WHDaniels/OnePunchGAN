from trainers.CycleTrainer import CycleTrainer
from trainers.GenTrainer import GenTrainer
from trainers.FinalTrainer import FinalTrainer
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='cycle')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--test_path', type=str, default=str(os.getcwd()) + '\\test')
parser.add_argument('--train_path', type=str, default=str(os.getcwd()) + '\\train')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
parser.add_argument('--decay_epochs', type=int, default=0, help='Number of epochs which the learning rate '
                                                                'decays linearly')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--multi_gpu', type=bool, default=False)

parser.add_argument('--gen_A2B', type=str, default='', help='Path to the A to B generator')
parser.add_argument('--gen_B2A', type=str, default='', help='Path to the B to A generator')
parser.add_argument('--dis_A', type=str, default='', help='Path to the A discriminator')
parser.add_argument('--dis_B', type=str, default='', help='Path to the B discriminator')

parser.add_argument('--gen_A2B_dict', type=str, default='',
                    help='path to the state_dict to be loaded into the A to B generator')
parser.add_argument('--gen_B2A_dict', type=str, default='',
                    help='path to the state_dict to be loaded into the B to A generator')
parser.add_argument('--dis_A_dict', type=str, default='',
                    help='path to the state_dict to be loaded into the A discriminator')
parser.add_argument('--dis_B_dict', type=str, default='',
                    help='path to the state_dict to be loaded into the B discriminator')

parser.add_argument('--save_path', type=str, default="E:\\MODELS\\")
parser.add_argument('--image_size', type=int, default=256, help='Size of image resize in training')
parser.add_argument('--results_path', type=str, default=str(os.getcwd()) + '\\results')
parser.add_argument('--output_path', type=str, default=str(os.getcwd()) + '\\output')
parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle data in data loader after each epoch')
parser.add_argument('--metrics', type=bool, default=False, help='Output loss versus number of epochs plot (per epoch)')

parser.add_argument('--critic_iters', type=int, default=2, help='Iterations to run critic for per run')

args = parser.parse_args()

if __name__ == "__main__":

    if args.model == "cycle" and args.mode == 'train':
        trainer = CycleTrainer(args)
        trainer.train()

    elif args.model == "cycle" and args.mode == 'test':
        trainer = CycleTrainer(args)
        trainer.test()

    elif args.model == "gen" and args.mode == 'train':
        trainer = GenTrainer(args)
        trainer.train()

    elif args.model == "gen" and args.mode == 'test':
        trainer = GenTrainer(args)
        trainer.test()

    elif 'final' in args.model and args.mode == 'train':
        trainer = FinalTrainer(args)
        trainer.train()

    elif 'final' in args.model and args.mode == 'test':
        trainer = FinalTrainer(args)
        trainer.test()

    else:
        print("Please enter a valid model: \'cycle\', \'gan\', \'gen\', \'final\', or \'final_with_wgangp\' "
              "along with a valid mode: \'train\' or \'test\'")
