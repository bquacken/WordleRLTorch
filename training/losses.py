import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from config import params

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

critic_loss_weight = params['critic_loss_weight']
entropy_loss_weight = params['entropy_weight']


def critic_loss_fn(returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return critic_loss_weight * nn.HuberLoss(reduction='mean')(returns, values)


def actor_loss_fn(actions_advantages: torch.Tensor,
                  policy_logits: torch.Tensor, char_logits) -> torch.Tensor:
    actions = actions_advantages[:, 0].type(torch.LongTensor).to(device)
    advantages = actions_advantages[:, 1].to(device)
    probs = nn.Softmax(dim=1)(policy_logits).to(device)
    char_probs =  nn.Softmax(dim=1)(char_logits).to(device)
    policy_loss = -(advantages * torch.log(probs)[list(range(len(actions))), actions]).mean()
    if params['char_logit']:
        entropy_loss = entropy_loss_weight * (-char_probs * torch.log(char_probs)).sum(dim=1).mean()
    else:
        entropy_loss = entropy_loss_weight * (-probs * torch.log(probs)).sum(dim=1).mean()
    return policy_loss , entropy_loss


