import torch
import torch.nn as nn
from config import params
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class StateAutoencoder(nn.Module):
    def __init__(self):
        super(StateAutoencoder, self).__init__()
        self.state_dim = params['state_dim']
        self.embed_dim = params['embed_dim']
        self.encode1 = nn.Linear(self.state_dim, 256)
        self.encode2 = nn.Linear(256, self.embed_dim)
        self.decode1 = nn.Linear(self.embed_dim, 256)
        self.decode2 = nn.Linear(256, self.state_dim)
        self.encoder = nn.Sequential(self.encode1, nn.ReLU(), self.encode2)
        self.decoder = nn.Sequential(self.decode1, nn.ReLU(), self.decode2, nn.Sigmoid())

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
