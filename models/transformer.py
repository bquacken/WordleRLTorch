import torch
import torch.nn as nn

from wordle.get_word_funcs import one_hot_words
from config import params
from models.a2c import compute_advantages
from training.losses import actor_loss_fn, critic_loss_fn

pi = torch.pi


def transformer_lr(epoch):
    if epoch < 6000:
        return 10
    else:
        return 1


class Head(nn.Module):
    def __init__(self, d_in, d_embed):
        super().__init__()
        self.d_in = d_in
        self.d_embed = d_embed
        self.q = nn.Linear(d_in, d_embed)
        self.k = nn.Linear(d_in, d_embed)
        self.v = nn.Linear(d_in, d_embed)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / self.d_embed ** 0.5
        attn = self.softmax(attn)
        out = torch.matmul(attn, v.unsqueeze(2)).squeeze(-1)
        return out


class MultiHead(nn.Module):
    def __init__(self, d_in, num_heads):
        super().__init__()
        self.d_in = d_in
        self.d_embed = d_in // num_heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head(d_in, self.d_embed) for _ in range(num_heads)])
        self.out = nn.Linear(self.d_embed * num_heads, d_in)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=1)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.lin1 = nn.Linear(d_embed, d_embed * 4)
        self.lin2 = nn.Linear(d_embed * 4, d_embed)

    def forward(self, x):
        out = self.lin1(x)
        out = nn.ReLU()(out)
        out = self.lin2(out)
        return out


class Block(nn.Module):
    def __init__(self, d_embed, num_heads):
        super().__init__()
        self.d_embed = d_embed
        self.num_heads = num_heads
        self.attn = MultiHead(d_embed, num_heads)
        self.ff = FeedForward(d_embed)
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ActorCriticTransformer(nn.Module):
    def __init__(self, mode: str = 'hard', dev: str = 'cpu'):
        super().__init__()
        assert mode in ['easy', 'hard']
        self.mode = mode
        self.device = torch.device(dev)

        self.d_input = params['state_dim']
        self.d_embed = 128
        self.num_heads = 4
        self.num_blocks = 2

        self.encoder = nn.Sequential(nn.Linear(self.d_input, self.d_embed), nn.ReLU())
        self.block = nn.Sequential(*[Block(self.d_embed, self.num_heads) for _ in range(self.num_blocks)])
        self.value = nn.Linear(self.d_embed, 1)
        self.policy = nn.Linear(self.d_embed, 26 * 5)

        self.num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.word_matrix = torch.Tensor(one_hot_words(self.mode)).to(self.device)

        if dev == 'cuda':
            self.encoder.cuda()
            self.block.cuda()
            self.value.cuda()
            self.policy.cuda()

        self.optim = torch.optim.RMSprop(self.parameters(), lr=params['transformer_lr'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, transformer_lr)

    def forward(self, x):
        x = self.encoder(x)
        x = self.block(x)
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
        logits, values = self.forward(states)
        actor_loss = actor_loss_fn(acts_advs, logits)
        critic_loss = critic_loss_fn(returns, values.squeeze())
        loss = actor_loss + critic_loss
        loss.backward()
        self.optim.step()
        self.scheduler.step()
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
