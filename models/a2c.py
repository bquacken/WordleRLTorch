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
        return 5
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
        self.d_embed = 256
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
        self.optim = torch.optim.Adam(self.parameters(), lr=params['mlp_lr'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, mlp_lr)

    def forward(self, x):
        x = self.head(x)
        value = self.value(x)
        policy = self.policy(x)
        policy = policy @ self.word_matrix.t()
        return policy, value

    def action_value(self, state: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(state)
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample([1])
        return action, value

    def character_logits(self, x):
        x = self.head(x)
        x = self.policy(x)
        return x

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
        logits, value = self(states)
        char_logits = self.character_logits(states)
        policy_loss, entropy_loss = actor_loss_fn(acts_advs, logits, char_logits)
        critic_loss = critic_loss_fn(returns, values)
        loss = policy_loss - entropy_loss + critic_loss
        loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), clip_value=10.0)
        self.optim.step()
        self.scheduler.step()
        return policy_loss.item(), entropy_loss.item(), critic_loss.item()
