from argparse import ArgumentParser
import torch

from training.embed_train import train_embedding

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=bool)
    parser.add_argument('--save', type=bool)
    parser.add_argument('--plot', type=bool)
    args = parser.parse_args()

    if args.resume is None:
        resume = False
    else:
        resume = True
    if args.save is None:
        save = False
    else:
        save = True
    losses = train_embedding(epochs=args.epochs, resume=resume, save=args.save, plot=args.plot)
