import torch
import os
from torch.utils.data import Dataset

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class StateDataset(Dataset):
    def __init__(self, string):
        if string == 'train':
            self.path = 'dataloader/data/embedding_state_data/train/'
            self.files = os.listdir(self.path)
        elif string == 'test':
            self.path = 'dataloader/data/embedding_state_data/test/'
            self.files = os.listdir(self.path)
        else:
            raise ValueError('Wrong Dataset')

    def __getitem__(self, item):
        file = self.files[item]
        file = torch.load(self.path + file).to(device, dtype=torch.float32)
        return file

    def __len__(self):
        return len(self.files)
