from argparse import ArgumentParser

from training.a2c_train import train_model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--mode', type=str, default='easy')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--bench', type=bool, default=True)

    args = parser.parse_args()


    train_model(epochs=args.epochs, mode=args.mode, model_str=args.model, resume=args.resume, save=args.save, bench=args.bench)

