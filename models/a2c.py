import torch
import torch.nn as nn
from config import params
from wordle.get_word_funcs import one_hot_words
from training.losses import actor_loss_fn, critic_loss_fn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device = torch.device('cpu')


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
    discount_factor = 0.99
    with torch.no_grad():
        advantages = torch.zeros(len(rewards))
        advantages[-1] = rewards[-1] - values[-1]
        returns = torch.zeros(len(rewards)).to(values.device)
        returns[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + (1 - dones[i]) * discount_factor * returns[i + 1]
        advantages = returns - values
        return returns, advantages


class Actor(nn.Module):
    """
    Advantage Actor Critic Model for Wordle:
    mode (str): 'easy' or 'hard'
    """

    def __init__(self, mode: str = 'hard', dev: str = 'cpu', fine_tune: str = False):
        super().__init__()
        self.mode = mode
        self.device = torch.device(dev)
        self.state_dim = params['state_dim']
        self.embed_dim = params['embed_dim']
        self.encoder = nn.Sequential(nn.Linear(self.state_dim, 256), nn.ReLU(), nn.Linear(256, self.embed_dim))
        self.encoder.load_state_dict(torch.load('models/model_weights/encoder_weights', map_location=self.device))
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.n_outputs = 130
        self.n_neurons = 512
        self.relu = nn.ReLU()
        self.policy1 = nn.Linear(self.embed_dim, self.n_neurons)
        self.policy2 = nn.Linear(self.n_neurons, self.n_neurons)
        self.policy3 = nn.Linear(self.n_neurons, self.n_outputs)
        if self.mode in ['easy', 'hard']:
            self.word_matrix = torch.Tensor(one_hot_words(self.mode)).to(self.device)
        else:
            raise Exception('Invalid Game Mode')
        self.optim = torch.optim.Adam(self.parameters(), lr=params['actor_lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=params['gamma'],
                                                                    patience=1500, min_lr=1e-9, verbose=True)

        self.encoder.to(self.device)
        for param in self.parameters():
            param.to(self.device)

        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, inputs: torch.Tensor):
        x = self.encoder(inputs)
        x = self.relu(self.policy1(x))
        x = self.relu(self.policy2(x))
        x = self.relu(self.policy3(x))
        x = self.word_matrix @ torch.t(x)
        return torch.t(x)

    def action(self, state: torch.Tensor):
        logits = self.forward(state)
        action = torch.distributions.Categorical(logits=logits).sample([1])
        return action

    def train_on_batch(self, memory):
        states = torch.stack(memory.states).to(self.device)
        actions = torch.Tensor(memory.actions).to(self.device)
        values = torch.Tensor(memory.values).to(self.device)
        dones = torch.tensor(memory.dones, dtype=torch.int8).to(self.device)
        rewards = torch.tensor(memory.rewards, dtype=torch.int8).to(self.device)

        returns, advantages = compute_advantages(rewards, values, dones)
        acts_advs = torch.zeros((len(actions), 2))
        acts_advs[:, 0] = actions
        acts_advs[:, 1] = advantages

        self.optim.zero_grad()
        logits = self.forward(states)
        actor_loss = actor_loss_fn(acts_advs, logits)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optim.step()
        self.scheduler.step(torch.mean(returns))
        return actor_loss.detach().cpu().numpy()


class Critic(nn.Module):
    """
    Advantage Actor Critic Model for Wordle:
    mode (str): 'easy' or 'hard'
    """

    def __init__(self, mode: str = 'hard', dev: str = 'cpu', fine_tune: bool = False):
        super().__init__()
        self.mode = mode
        self.device = torch.device(dev)
        self.state_dim = params['state_dim']
        self.embed_dim = params['embed_dim']
        self.encoder = nn.Sequential(nn.Linear(self.state_dim, 256), nn.ReLU(), nn.Linear(256, self.embed_dim))
        self.encoder.load_state_dict(torch.load('models/model_weights/encoder_weights', map_location=self.device))
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.n_outputs = 130
        self.n_neurons = 512
        self.relu = nn.ReLU()
        self.value1 = nn.Linear(self.embed_dim, self.n_neurons)
        self.value2 = nn.Linear(self.n_neurons, self.n_neurons)
        self.value3 = nn.Linear(self.n_neurons, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=params['critic_lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=params['gamma'],
                                                                    patience=1500, min_lr=1e-8, verbose=True)
        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder.to(self.device)
        for param in self.parameters():
            param.to(self.device)


    def forward(self, inputs: torch.Tensor):
        x = self.encoder(inputs)
        x = self.relu(self.value1(x))
        x = self.relu(self.value2(x))
        value = self.value3(x)
        return value

    def train_on_batch(self, memory):
        states = torch.stack(memory.states).to(self.device)
        actions = torch.Tensor(memory.actions).to(self.device)
        values = torch.Tensor(memory.values).to(self.device)
        dones = torch.tensor(memory.dones, dtype=torch.int8).to(self.device)
        rewards = torch.tensor(memory.rewards, dtype=torch.int8).to(self.device)

        returns, advantages = compute_advantages(rewards, values, dones)
        acts_advs = torch.zeros((len(actions), 2))
        acts_advs[:, 0] = actions
        acts_advs[:, 1] = advantages
        returns = returns.view(len(rewards), 1)

        self.optim.zero_grad()
        values = self.forward(states)
        critic_loss = critic_loss_fn(returns, values)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optim.step()
        self.scheduler.step(torch.mean(returns))
        return critic_loss.detach().cpu().numpy()


"""class A2C(nn.Module):
    """
# Advantage Actor Critic Model for Wordle:
# mode (str): 'easy' or 'hard'
"""

def __init__(self, mode: str = 'hard'):
    super().__init__()
    self.mode = mode
    self.state_dim = params['state_dim']
    self.embed_dim = params['embed_dim']
    self.encoder = nn.Sequential(nn.Linear(self.state_dim, 128), nn.ReLU(), nn.Linear(128, self.embed_dim))
    self.encoder.load_state_dict(torch.load('models/model_weights/encoder_weights', map_location=device))
    for param in self.encoder.parameters():
        param.requires_grad = False

    self.n_outputs = 130
    self.n_neurons = 64
    self.relu = nn.ReLU()
    self.policy1 = nn.Linear(self.embed_dim, self.n_neurons)
    self.policy2 = nn.Linear(self.n_neurons, self.n_outputs)
    self.value1 = nn.Linear(self.embed_dim, self.n_neurons)
    self.value2 = nn.Linear(self.n_neurons, 1)
    if self.mode in ['easy', 'hard']:
        self.word_matrix = torch.Tensor(one_hot_words(self.mode))
    else:
        raise Exception('Invalid Game Mode')

def forward(self, inputs: torch.Tensor):
    x = self.encoder(inputs)
    actor = self.relu(self.policy1(x))
    actor = self.relu(self.policy2(actor))
    actor = self.word_matrix @ torch.t(actor)
    value = self.relu(self.value1(x))
    value = self.value2(value)
    return value, torch.t(actor)

def action_value(self, state: torch.Tensor):
    value, logits = self.forward(state)
    action = torch.distributions.Categorical(logits=logits).sample([1])
    return value, action"""
