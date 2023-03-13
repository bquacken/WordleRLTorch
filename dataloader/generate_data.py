import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import torch

from wordle import Environment
from wordle import get_word_funcs

total_words = get_word_funcs.get_words(False)


def get_index(word):
    arr = np.where(total_words == word)[0]
    if len(arr) == 1:
        return np.uint16(arr[0])
    else:
        return np.uint16(0)


def generate_game_data():
    print('Generating game data...')
    num_picks = 200
    df = pd.read_csv('dataloader/data/wordle_random.csv')
    game_repeats = df['solution'].value_counts()[0]
    total = len(df)
    df.sort_values('solution', inplace=True, ignore_index=True)

    indices = []
    for i in range(int(total / game_repeats)):
        indices += list(np.random.choice(range(game_repeats), num_picks) + game_repeats * i)

    df = df.loc[indices]

    solution = list(df['solution'])
    attempt0 = list(df['attempt_0'])
    attempt1 = list(df['attempt_1'])
    attempt2 = list(df['attempt_2'])
    attempt3 = list(df['attempt_3'])
    attempt4 = list(df['attempt_4'])
    attempt5 = list(df['attempt_5'])
    n = len(solution)

    with Pool(6) as pool:
        solution = np.array(list(pool.map(get_index, solution))).reshape((n, 1))
        attempt0 = np.array(list(pool.map(get_index, attempt0))).reshape((n, 1))
        attempt1 = np.array(list(pool.map(get_index, attempt1))).reshape((n, 1))
        attempt2 = np.array(list(pool.map(get_index, attempt2))).reshape((n, 1))
        attempt3 = np.array(list(pool.map(get_index, attempt3))).reshape((n, 1))
        attempt4 = np.array(list(pool.map(get_index, attempt4))).reshape((n, 1))
        attempt5 = np.array(list(pool.map(get_index, attempt5))).reshape((n, 1))

    games = np.hstack((solution, attempt0, attempt1, attempt2, attempt3, attempt4, attempt5))

    np.save('dataloader/data/wordle_np_games', games)

    return games


def generate_state_data():
    if os.path.exists('dataloader/data/wordle_np_games.npy'):
        games = np.load('dataloader/data/wordle_np_games.npy')
    else:
        games = generate_game_data()
    print('Generating state data...')

    env = Environment.Environment()
    states = env.state
    n = 100000
    temp_states = np.zeros((n, 448), dtype=np.uint8)
    j = 0
    for num in tqdm(range(games.shape[0])):
        env.reset(total_words[games[num, 0]])
        if np.random.uniform() < 0.05:
            if j < n:
                temp_states[j, :] = env.state.copy()
                j += 1
            else:
                j = 0
                states = np.vstack((states, temp_states))
                temp_states = np.zeros((n, 448), dtype=np.uint8)
                temp_states[j, :] = env.state.copy()
                j += 1
        i = 1
        while not env.wordle.over:
            if np.random.uniform() <= 0.65:
                env.step(games[num, i])
                i += 1
            else:
                env.step(np.random.choice(len(total_words)))
                i += 1

            if j < n:
                if not env.wordle.over:
                    temp_states[j, :] = env.state.copy()
                j += 1
            else:
                if not env.wordle.over:
                    j = 0
                    states = np.vstack((states, temp_states))
                    temp_states = np.zeros((n, 448), dtype=np.uint8)
                    temp_states[j, :] = env.state.copy()
                    j += 1
    ind = []
    for j in tqdm(range(states.shape[0])):
        if not (states[j, :] == 0).all():
            ind.append(j)
    states = states[ind, :]
    np.random.shuffle(states)
    np.save('dataloader/data/state_data', states)
    return states


def batch_state_data():
    if os.path.exists('dataloader/data/state_data.npy'):
        states = np.load('dataloader/data/state_data.npy')
    else:
        states = generate_state_data()

    print('Batching state data...')

    batch_size = 2048
    train = 0.6
    test = 0.4

    num_train_iter = int(train * states.shape[0] // batch_size)
    num_test_iter = int(test * states.shape[0] // batch_size)
    test_start = num_train_iter * batch_size

    for num in tqdm(range(num_train_iter)):
        torch.save(torch.tensor(states[batch_size * num: batch_size * (num + 1), :]),
                   f'dataloader/data/embedding_state_data/train/batch_{num}')
    for num in tqdm(range(num_test_iter)):
        torch.save(torch.tensor(states[test_start + batch_size * num: test_start + batch_size * (num + 1), :]),
                   f'dataloader/data/embedding_state_data/test/batch_{num}')


if __name__ == '__main__':
    batch_state_data()