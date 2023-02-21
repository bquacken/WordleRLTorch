import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

critic_loss_weight = 0.5
actor_loss_weight = 1
entropy_loss_weight = 0.01
eps = np.finfo(np.float32).eps.item()


def critic_loss_fn(returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return nn.HuberLoss(reduction='mean')(returns, values)


def actor_loss_fn(actions_advantages: torch.Tensor, policy_logits: torch.Tensor) -> torch.Tensor:
    actions = actions_advantages[:, 0].type(torch.LongTensor).to(device)
    advantages = actions_advantages[:, 1].to(device)
    probs = nn.Softmax(dim=1)(policy_logits).to(device)
    policy_loss = -(advantages * torch.log(probs)[list(range(len(actions))), actions]).mean()
    entropy_loss = (-probs * torch.log(probs)).sum(dim=1).mean()
    return actor_loss_weight * policy_loss - entropy_loss_weight * entropy_loss


class EmbeddingLoss(_Loss):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')

    def forward(self, inputs, targets):
        batch = inputs.size(0)
        output_loss = self.cos_loss(inputs[:, :6], targets[:, :6], torch.ones(batch))
        output_loss += self.cos_loss(inputs[:, 6:32], targets[:, 6:32], torch.ones(batch))
        output_loss += self.cos_loss(inputs[:, 32:58], targets[:, 32:58], torch.ones(batch))
        for j in range(5):
            output_loss += self.cos_loss(inputs[:, 58 + 26 * 3 * j : 58 + 26 * 3 * (j+1)],
                                         targets[:, 58 + 26 * 3 * j : 58 + 26 * 3 * (j+1)],
                                         torch.ones(batch))
        return output_loss
