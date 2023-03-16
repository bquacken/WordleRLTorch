import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from config import params
from models.a2c import Actor, Critic
from wordle.Environment import Environment
from training.simulate_games import ParallelEnvironments
from training.benchmark import benchmark_model
from wordle.get_word_funcs import get_words
import faulthandler

if torch.cuda.is_available():
    device = torch.device('cuda')
    dev = 'cuda'
else:
    device = torch.device('cpu')
    dev = 'cpu'
cpu = torch.device('cpu')
device = torch.device('cpu')


def train_model(epochs, mode='easy', resume=False, save=False, bench=True, fine_tune=False, clip=False):
    actor = Actor(mode, dev, fine_tune=fine_tune)
    critic = Critic(mode, dev, fine_tune=fine_tune)
    actor.clip = clip
    critic.clip = clip

    if dev == 'cuda':
        actor.cuda()
        critic.cuda()

    batch_size = params['batch_size']
    if resume:
        print('Loading weights...')
        actor.load_state_dict(torch.load(f'models/model_weights/actor_{mode}', map_location=device))
        actor.optim.load_state_dict(torch.load(f'models/model_weights/actor_optim_{mode}', map_location=device))
        critic.load_state_dict(torch.load(f'models/model_weights/critic_{mode}', map_location=device))
        critic.optim.load_state_dict(torch.load(f'models/model_weights/critic_optim_{mode}', map_location=device))

    if fine_tune:
        print('Finetuning...')
        for param in actor.encoder.parameters():
            param.requires_grad = True
        for param in critic.encoder.parameters():
            param.requires_grad = True

    total_words = get_words()

    if mode == 'easy':
        answers = total_words[:params['easy_mode']]
    else:
        answers = total_words[:2309]

    para_env = ParallelEnvironments(params['parallel_workers'], mode)

    rewards_list = []
    first_turn_rewards = []
    actor_loss_list = []
    critic_loss_list = []

    faulthandler.enable()
    for ep in tqdm(range(1, epochs + 1)):
        para_env.load_weights(actor.state_dict(), critic.state_dict())
        memory, rewards, first_rewards = para_env.parallel_simulate(batch_size, answers)

        actor_loss = actor.train_on_batch(memory)
        critic_loss = critic.train_on_batch(memory)

        actor_loss_list.append(actor_loss)
        critic_loss_list.append(critic_loss)
        rewards_list += rewards
        first_turn_rewards += first_rewards

        if ep % 2000 == 0:
            torch.save(actor.state_dict(), f'models/model_weights/actor_temp_{int((ep // 2000) % 6)}')
            torch.save(critic.state_dict(), f'models/model_weights/critic_temp_{int((ep // 2000) % 6)}')
            print(f'benchmark {ep // 2000}')
            actor.cpu()
            actor.word_matrix = actor.word_matrix.cpu()
            benchmark_model(actor, answers)
            actor.cuda()
            actor.word_matrix = actor.word_matrix.cuda()

    env = Environment()
    env.reset(np.random.choice(answers, 1)[0])
    actor.cpu()
    actor.word_matrix = actor.word_matrix.cpu()
    critic.cpu()
    while not env.wordle.over:
        with torch.no_grad():
            state = torch.Tensor(env.state)
            action = actor.action(state[None, :])
            action = int(action[0][0])
            env.step(action)

    print('Answer: ', env.wordle.answer)
    print('Guesses: ', env.wordle.guesses)

    if bench:
        benchmark_model(actor, answers)

    rewards_list = np.array(rewards_list)
    avg_rewards = np.array(np.convolve(rewards_list, np.ones(100) / 100)[100:-100], dtype=np.float16)
    first_turn_rewards = np.array(first_turn_rewards)
    avg_first_rewards = np.array(np.convolve(first_turn_rewards, np.ones(100) / 100)[100:-100], dtype=np.float16)
    np.save('avg_rewards', avg_rewards)
    np.save('avg_first_rewards', avg_first_rewards)

    now = datetime.now()
    now = now.strftime("%m&d&Y_&H:%M")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.set_title('Actor Loss')
    ax1.plot(actor_loss_list)
    ax2.plot(critic_loss_list)
    ax2.set_title('Critic Loss')
    ax3.set_title('Average Rewards')
    ax3.plot(avg_rewards)
    ax4.set_title('Average Rewards from first turn')
    ax4.plot(avg_first_rewards)
    fig.savefig(f'training/plots/rewards_losses_{now}.jpeg')
    plt.show()

    if not save:
        save = input('Do you want to save models weights? Y for yes, N for No: ')
        if save.lower() == 'y':
            print('Saving Model Weights...')
            torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}')
            torch.save(actor.optim.state_dict(), f'models/model_weights/actor_optim_{mode}')
            torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}')
            torch.save(critic.optim.state_dict(), f'models/model_weights/critic_optim_{mode}')
    elif save:
        print('Saving Model Weights...')
        torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}')
        torch.save(actor.optim.state_dict(), f'models/model_weights/actor_optim_{mode}')
        torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}')
        torch.save(critic.optim.state_dict(), f'models/model_weights/critic_optim_{mode}')

    return actor_loss_list, critic_loss_list
