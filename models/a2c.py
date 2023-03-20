import torch
import torch.nn as nn
from config import params
from wordle.get_word_funcs import one_hot_words
from training.losses import actor_loss_fn, critic_loss_fn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def mlp_lr(epoch):
    if epoch < 4000:
        return 1
    else:
        return 1


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


class ActorCritic(nn.Module):
    def __init__(self, mode: str = 'hard', dev: str = 'cpu'):
        super().__init__()
        assert mode in ['easy', 'hard']
        self.mode = mode
        self.device = torch.device(dev)

        self.d_input = params['state_dim']
        self.d_embed = 512
        self.head = nn.Sequential(nn.Linear(self.d_input, self.d_embed), nn.Tanh(),
                                  nn.Linear(self.d_embed, self.d_embed), nn.Tanh(),
                                  nn.Linear(self.d_embed, self.d_embed), nn.Tanh())

        self.value = nn.Sequential(nn.Linear(self.d_embed, self.d_embed),
                                   nn.Tanh(),
                                   nn.Linear(self.d_embed, 1))

        self.policy = nn.Sequential(nn.Linear(self.d_embed, self.d_embed),
                                    nn.Tanh(),
                                    nn.Linear(self.d_embed, 130))

        self.num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.word_matrix = torch.Tensor(one_hot_words(self.mode)).to(self.device)

        self.dev = torch.device(dev)

        if dev == 'cuda':
            self.head.cuda()
            self.value.cuda()
            self.policy.cuda()
        self.optim = torch.optim.RMSprop(self.parameters(), lr=params['mlp_lr'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, mlp_lr)

    def forward(self, x):
        x = self.head(x)
        value = self.value(x)
        policy = self.policy(x)
        policy = policy @ self.word_matrix.t()
        return policy, value

    def action_value(self, state: torch.Tensor):
        logits, value = self.forward(state)
        action = torch.distributions.Categorical(logits=logits).sample([1])
        return action, value

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
        logits, value = self.forward(states)
        actor_loss = actor_loss_fn(acts_advs, logits)
        critic_loss = critic_loss_fn(returns, values)
        loss = actor_loss + critic_loss
        loss.backward()
        self.optim.step()
        self.scheduler.step()
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
