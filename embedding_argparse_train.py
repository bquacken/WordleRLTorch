from argparse import ArgumentParser
import torch

from training.embed_train import train_embedding

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    args = parser.parse_args()


    losses = train_embedding(epochs=args.epochs, resume=args.resume, save=args.save, plot=args.plot, test=args.test)
