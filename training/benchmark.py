import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from wordle import Environment


def benchmark_model(actor: nn.Module, answers):
    env = Environment.Environment()
    wins = 0
    scores = []
    actor.eval()
    for ans in tqdm(answers):
        env.reset(ans)
        while not env.wordle.over:
            action = actor.action(torch.Tensor(env.state)[None, :])
            action = int(action[0][0])
            env.step(action)
        if env.wordle.win:
            wins += 1
            scores.append(env.num_guesses)
    avg_score = np.mean(scores)
    print(f'Total Wins out of {len(answers)} Games: {wins}')
    print(f'Average Number of Guesses per Win: {avg_score:02f}')
    return wins, avg_score
