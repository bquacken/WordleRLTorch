import torch
import numpy as np
from models.a2c import ActorCritic
from models.transformer import ActorCriticTransformer
from training.Memory import Memory
from wordle.Environment import Environment
from joblib import Parallel, delayed, parallel_backend

device = torch.device('cpu')


class ParallelEnvironments:
    def __init__(self, num_workers: int, model_str: str, mode: str = 'hard'):
        self.num_workers = num_workers
        if model_str == 'mlp':
            self.models = [ActorCritic(mode, dev='cpu') for _ in range(self.num_workers)]
        elif model_str == 'transformer':
            self.models = [ActorCriticTransformer(mode, dev='cpu') for _ in range(self.num_workers)]
        self.rand_threshold = 0.05 if mode =='easy' else 0.15

    def load_weights(self, model_weights):
        for i in range(self.num_workers):
            self.models[i].load_state_dict(model_weights)
            self.models[i].cpu()

    def simulate(self, num_games, model, answers):
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
            if losing_word_list == [] or rand > self.rand_threshold:
                env.reset(np.random.choice(answers, 1)[0])
            else:
                env.reset(losing_word_list.pop(0))
            while not env.wordle.over:
                with torch.no_grad():
                    state = torch.Tensor(env.state)
                    action, value = model.action_value(state[None, :])
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

    def single_simulate(self, num_games, answers):
        [memory, rewards_list, first_turn_rewards] = self.simulate(num_games, self.models[0], answers)
        return memory, rewards_list, first_turn_rewards

    def parallel_simulate(self, num_games, answers):
        games_per_worker = int(num_games / self.num_workers)

        with parallel_backend('loky', inner_max_num_threads=3):
            result_list = Parallel(n_jobs=self.num_workers, verbose=0) \
                (delayed(self.simulate)(games_per_worker, self.models[i], answers) for i in
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
