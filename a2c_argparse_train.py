from argparse import ArgumentParser

from training.a2c_train import train_model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--mode', type=str, default='easy')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--bench', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--clip', type=bool, default=False)
    args = parser.parse_args()


    train_model(epochs=args.epochs, mode=args.mode, resume=args.resume, save=args.save, bench=args.bench, fine_tune=args.finetune, clip=args.clip)

