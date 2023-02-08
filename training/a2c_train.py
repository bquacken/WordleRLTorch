import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import params
from models.a2c import Actor, Critic
from wordle.Environment import Environment
from training.simulate_games import ParallelEnvironments
from training.benchmark import benchmark_model
from wordle.get_word_funcs import get_words

if torch.cuda.is_available():
    device = torch.device('cuda')
    dev = 'cuda'
else:
    device = torch.device('cpu')
    dev = 'cpu'
cpu = torch.device('cpu')
device = torch.device('cpu')


def train_model(epochs, mode='easy', resume=False, save=False, bench=True):
    actor = Actor(mode, dev)
    critic = Critic(mode, dev)

    if dev == 'cuda':
        actor.cuda()
        critic.cuda()


    batch_size = params['batch_size']
    if resume:
        print('Loading weights...')
        actor.load_state_dict(torch.load(f'models/model_weights/actor_{mode}', map_location=device))
        critic.load_state_dict(torch.load(f'models/model_weights/critic_{mode}', map_location=device))

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

    for _ in tqdm(range(epochs)):
        para_env.load_weights(actor.state_dict(), critic.state_dict())
        memory, rewards, first_rewards = para_env.parallel_simulate(batch_size, answers)

        actor_loss = actor.train_on_batch(memory)
        critic_loss = critic.train_on_batch(memory)

        actor_loss_list.append(actor_loss)
        critic_loss_list.append(critic_loss)
        rewards_list += rewards
        first_turn_rewards += first_rewards
        if epochs % 4000 == 0:
            torch.save(actor.state_dict(), f'models/model_weights/actor_temp_{int(epochs // 4000)}')
            torch.save(critic.state_dict(), f'models/model_weights/critic_temp_{int(epochs // 4000)}')



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
    avg_rewards = np.convolve(rewards_list, np.ones(100)/100)[100:-100]
    first_turn_rewards = np.array(first_turn_rewards)
    avg_first_rewards = np.convolve(first_turn_rewards, np.ones(100)/100)[100:-100]


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.set_title('Actor Loss')
    ax1.plot(actor_loss_list)
    ax2.plot(critic_loss_list)
    ax2.set_title('Critic Loss')
    ax3.set_title('Average Rewards')
    ax3.plot(avg_rewards)
    ax4.set_title('Average Rewards from first turn')
    ax4.plot(avg_first_rewards)
    plt.show()

    if not save:
        save = input('Do you want to save models weights? Y for yes, N for No: ')
        if save.lower() == 'y':
            print('Saving Model Weights...')
            torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}')
            torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}')
    elif save:
        print('Saving Model Weights...')
        torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}')
        torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}')

    return actor_loss_list, critic_loss_list


"""
def train_model(epochs,
                batch_size: int = 1000,
                mode: str = 'hard',
                resume: bool = False,
                save: bool = False,
                bench: bool = False):
    total_words = get_words()
    if mode == 'hard':
        answers = total_words[:2309]
    else:
        answers = total_words[:params['easy_mode']]
    losses = []
    actor_losses = []
    critic_losses = []
    losing_word_list = []
    first_turn_rewards = []
    rewards_list = []
    memory = Memory()
    memory.clear()
    env = Environment.Environment()
    # model = A2C(mode)
    actor = Actor()
    critic = Critic()
    if resume:
        print('Loading Model Weights...')
        actor.load_state_dict(torch.load(f'models/model_weights/actor_easy'))
        critic.load_state_dict(torch.load(f'models/model_weights/critic_easy'))
        print('Done')
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optim, step_size=1000, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optim, step_size=1000, gamma=0.5)
    for game in tqdm(range(1, epochs + 1)):
        total_training_rewards = 0
        rand = np.random.uniform(0, 1, 1)[0]
        if losing_word_list == [] or rand > 0.05:
            env.reset(np.random.choice(answers, 1)[0])
        else:
            env.reset(losing_word_list.pop(0))
        while not env.wordle.over:
            with torch.no_grad():
                state = torch.Tensor(env.state)
                action = actor.action(state[None, :])
                value = critic(state[None, :])
                action = int(action[0][0])
                next_state, reward, done = env.step(action)
                memory.add(state, action, value.cpu().numpy()[0][0], done, reward)
                if env.num_guesses == 1:
                    first_turn_rewards.append(reward)
                total_training_rewards += reward
        rewards_list.append(total_training_rewards)
        if not env.wordle.win:
            losing_word_list.append(env.wordle.answer)
        if game % batch_size == 0:
            states = torch.stack(memory.states).to(device)
            actions = torch.Tensor(memory.actions).to(device)
            values = torch.Tensor(memory.values).to(device)
            dones = torch.tensor(memory.dones, dtype=torch.int8).to(device)
            rewards = torch.tensor(memory.rewards, dtype=torch.int8).to(device)

            returns, advantages = compute_advantages(rewards, values, dones)
            acts_advs = torch.zeros((len(actions), 2))
            acts_advs[:, 0] = actions
            acts_advs[:, 1] = advantages
            returns = returns.view(len(rewards), 1)

            actor_optim.zero_grad()
            critic_optim.zero_grad()
            values = critic(states)
            logits = actor(states)
            critic_loss = critic_loss_fn(returns, values)
            actor_loss = actor_loss_fn(acts_advs, logits)
            loss = actor_loss + critic_loss
            loss.backward()
            actor_optim.step()
            critic_optim.step()
            actor_losses.append(actor_loss.detach().cpu().numpy())
            critic_losses.append(critic_loss.detach().cpu().numpy())
            memory.clear()
            actor_scheduler.step()
            critic_scheduler.step()
        if game % 1000000 == 0:
            benchmark_model(actor, answers)
            print('Saving Model Weights...')
            torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}_temp_{game // 1000000}')
            torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}_temp_{game // 1000000}')

    env.reset(np.random.choice(answers, 1)[0])
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

    rewards = np.zeros(len(rewards_list) - 100)
    first_rewards = np.zeros(len(first_turn_rewards) - 100)
    for i in range(100, len(rewards_list)):
        rewards[i - 100] = np.mean(rewards_list[i - 100:i])
    for i in range(100, len(first_turn_rewards)):
        first_rewards[i - 100] = np.mean(first_turn_rewards[i - 100:i])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.set_title('Actor Loss')
    ax1.plot(actor_losses)
    ax2.plot(critic_losses)
    ax2.set_title('Critic Loss')
    ax3.set_title('Average Rewards')
    ax3.plot(rewards)
    ax4.set_title('Average Rewards from first turn')
    ax4.plot(first_rewards)
    plt.show()

    if not save:
        save = input('Do you want to save models weights? Y for yes, N for No: ')
        if save.lower() == 'y':
            print('Saving Model Weights...')
            torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}')
            torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}')
    elif save:
        print('Saving Model Weights...')
        torch.save(actor.state_dict(), f'models/model_weights/actor_{mode}')
        torch.save(critic.state_dict(), f'models/model_weights/critic_{mode}')

    return losses
"""
