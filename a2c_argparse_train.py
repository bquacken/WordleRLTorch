from argparse import ArgumentParser

from training.a2c_train import train_model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--resume', type=bool)
    parser.add_argument('--save', type=bool)
    parser.add_argument('--bench', type=bool)
    args = parser.parse_args()
    if args.mode is None:
        mode = 'hard'
    else:
        mode = 'easy'
    if args.resume is None:
        resume = False
    else:
        resume = True
    if args.save is None:
        save = False
    else:
        save = True

    train_model(epochs=args.epochs, mode=mode, resume=resume, save=args.save, bench=args.bench)

