import torch
import numpy as np
from models.a2c import Actor, Critic
from training.Memory import Memory
from wordle.Environment import Environment
from joblib import Parallel, delayed

device = torch.device('cpu')

class ParallelEnvironments:
    def __init__(self, num_workers: int, mode: str = 'hard'):
        self.num_workers = num_workers
        self.actors = [Actor(mode, dev='cpu') for _ in range(self.num_workers)]
        self.critics = [Critic(mode, dev='cpu') for _ in range(self.num_workers)]


    def load_weights(self, actor_weights, critic_weights):
        # actor_weights.to(torch.device('cpu'))
        # critic_weights.to(torch.device('cpu'))
        for i in range(self.num_workers):
            self.actors[i].load_state_dict(actor_weights)
            self.critics[i].load_state_dict(critic_weights)
            self.actors[i].cpu()
            self.critics[i].cpu()

    def simulate(self, num_games, actor, critic, answers):
        np.random.seed()
        memory = Memory()
        memory.clear()
        env = Environment()
        losing_word_list = []
        first_turn_rewards = []
        rewards_list = []
        for _ in range(num_games):
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
        return [memory, rewards_list, first_turn_rewards]

    def parallel_simulate(self, num_games, answers):
        games_per_worker = int(num_games / self.num_workers)
        # input_args = [[games_per_worker, self.actors[i], self.critics[i], answers] for i in range(self.num_workers)]
        # pool = Pool(processes=self.num_workers)
        # memory_list = pool.starmap(self.simulate, iterable=iter(input_args))
        result_list = Parallel(n_jobs=self.num_workers, verbose=0) \
            (delayed(self.simulate)(games_per_worker, self.actors[i], self.critics[i], answers) for i in
             range(self.num_workers))
        memory_list = []
        reward_list = []
        first_turn_rewards = []
        for result in result_list:
            memory_list.append(result[0])
            reward_list += result[1]
            first_turn_rewards += result[2]
        main_memory = Memory()
        for mem in memory_list:
            main_memory.states += mem.states
            main_memory.actions += mem.actions
            main_memory.values += mem.values
            main_memory.dones += mem.dones
            main_memory.rewards += mem.rewards
        return main_memory, reward_list, first_turn_rewards
