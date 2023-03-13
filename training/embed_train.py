import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.classification import MultilabelAccuracy
from training.losses import EmbeddingLoss
from torch.utils.data import DataLoader
from models.embedding import StateAutoencoder
from dataloader.dataloader import StateDataset





if torch.cuda.is_available():
    # Desktop use
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    workers = 0
else:
    # Laptop use
    device = torch.device('cpu')
    workers = 1


def train_embedding(epochs: int = 10,
                    resume: bool = False,
                    save: bool = True,
                    plot: bool = False,
                    test: bool = True):
    model = StateAutoencoder()
    if resume:
        print('Loading weights...')
        model.load_state_dict(torch.load('models/model_weights/autoencoder_weights'))
    train_ds = StateDataset('train')
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', factor=0.1,
                                                           patience=3, min_lr=1e-9, verbose=True)
    train_dl = DataLoader(train_ds, batch_size=None)
    loss_fn = EmbeddingLoss()
    losses = []
    accuracy_fn = MultilabelAccuracy(448, threshold=0.5, average='none').to(device)
    for ep in range(epochs):
        accuracy = []
        epoch_losses = []
        print(f'Training Epoch {ep}:')
        for file in tqdm(train_dl):
            model.zero_grad()
            encoded_state = model(file)
            loss = loss_fn(encoded_state, file).mean()
            loss.backward()
            adam.step()
            losses.append(loss.detach().cpu().numpy())
            epoch_losses.append(losses[-1])
            accuracy.append(accuracy_fn(encoded_state, file).detach().cpu().mean())
        print('Accuracy: ', torch.mean(torch.Tensor(accuracy)))
        scheduler.step(torch.mean(torch.Tensor(np.array(losses[-400:]))))
    if save:
        print('Saving Weights')
        torch.save(model.encoder.state_dict(), 'models/model_weights/encoder_weights')
        torch.save(model.decoder.state_dict(), 'models/model_weights/decoder_weights')
        torch.save(model.state_dict(), 'models/model_weights/autoencoder_weights')

    if plot:
        avg_losses = np.convolve(losses, np.ones(100)/100)[100:-100]
        plt.plot(avg_losses)
        plt.title('Average Loss')
        plt.show()

    if test:
        test_ds = StateDataset('test')
        test_dl = DataLoader(test_ds, batch_size=None, num_workers=workers,
                          generator=torch.Generator(device='cuda'))
        test_accuracy = []
        model.eval()
        for file in tqdm(test_dl):
            encoded_state = model(file)
            test_accuracy.append(accuracy_fn(encoded_state, file).detach().cpu().mean())
        print('Test Accuracy: ', torch.mean(torch.Tensor(accuracy)))