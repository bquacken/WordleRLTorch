import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import wandb

from config import params
from models.a2c import ActorCritic
from models.transformer import ActorCriticTransformer
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


def train_model(epochs,
                mode='easy',
                model_str='mlp',
                resume=False,
                save=False,
                bench=True):
    """
    Parameters
    ----------
    :param epochs: int
    :   Number of epochs to train for
    :param mode: str
    :   'easy' or 'hard'
    :param model_str:
    :   'mlp' or 'transformer'
    :param resume:
    :   Whether to resume training from a previous checkpoint
    :param save:
    :   Whether to save the model weights after training
    :param bench:
    :   Whether to benchmark the model after training
    :return: None
    """

    if model_str == 'mlp':
        model = ActorCritic(mode, dev)
    elif model_str == 'transformer':
        model = ActorCriticTransformer(mode, dev)

    batch_size = params['batch_size']
    if resume:
        print('Loading weights...')
        model.load_state_dict(torch.load(f'models/model_weights/{model_str}_{mode}', map_location=device))

    total_words = get_words()

    if mode == 'easy':
        answers = total_words[:params['easy_mode']]
    else:
        answers = total_words[:2309]

    para_env = ParallelEnvironments(6, model_str, mode)

    rewards_list = []
    first_turn_rewards = []
    actor_loss_list = []
    critic_loss_list = []

    params['model'] = model_str
    params['resume'] = resume
    wandb.init(project='WordleRL', config=params)

    faulthandler.enable()
    for ep in tqdm(range(1, epochs + 1)):
        para_env.load_weights(model.state_dict())
        memory, rewards, first_rewards = para_env.parallel_simulate(batch_size, answers)

        policy_loss, entropy_loss, critic_loss = model.train_on_batch(memory)

        wandb.log({
            'epoch': ep,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'critic_loss': critic_loss
        })
        actor_loss_list.append(policy_loss - entropy_loss)
        critic_loss_list.append(critic_loss)
        rewards_list += rewards
        first_turn_rewards += first_rewards

        if ep % 2000 == 0:
            torch.save(model.state_dict(), f'models/model_weights/{model_str}_temp_{int((ep // 2000) % 6)}')
            print(f'benchmark {ep // 2000}')
            model.cpu()
            model.word_matrix = model.word_matrix.cpu()
            wins, avg_score = benchmark_model(model, answers)
            wandb.log({'epoch': ep, 'wins': wins, 'avg_score': avg_score})
            model.cuda()
            model.word_matrix = model.word_matrix.cuda()

    env = Environment()
    env.reset(np.random.choice(answers, 1)[0])
    model.cpu()
    model.word_matrix = model.word_matrix.cpu()

    while not env.wordle.over:
        with torch.no_grad():
            state = torch.Tensor(env.state)
            action, _ = model.action_value(state[None, :])
            action = int(action[0][0])
            env.step(action)

    print('Answer: ', env.wordle.answer)
    print('Guesses: ', env.wordle.guesses)

    if bench:
        benchmark_model(model, answers)

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
    fig.savefig(f'training/plots/{model_str}_rewards_losses_{now}.jpeg')
    # plt.show()

    if not save:
        save = input('Do you want to save models weights? Y for yes, N for No: ')
        if save.lower() == 'y':
            print('Saving Model Weights...')
            torch.save(model.state_dict(), f'models/model_weights/{model_str}_{mode}')

    elif save:
        print('Saving Model Weights...')
        torch.save(model.state_dict(), f'models/model_weights/{model_str}_{mode}')

    return
