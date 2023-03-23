import numpy as np
from wordle.Wordle import Wordle
from wordle.get_word_funcs import get_words
from config import params


class Environment:
    reward_win = 10
    reward_lose = -10

    def __init__(self, answer=None):
        self.score = 0
        self.rewards = [0]
        self.total_words = get_words()
        self.wordle = Wordle(answer)
        self.num_guesses = 0
        self.state = np.zeros(params['state_dim'], dtype=np.uint8)
        self.state[0] = 1
        self.guess_to_state()
        self.action_space = np.array(range(len(self.total_words)))
        self.bits_info = np.load('wordle/data/InitialStrategy.npy')

    def reset(self, answer=None):
        self.score = 0
        self.rewards = [0]
        self.wordle.reset(answer)
        self.num_guesses = 0
        self.state = np.zeros(params['state_dim'], dtype=np.uint8)
        self.guess_to_state()
        self.action_space = np.array(range(len(self.total_words)))

    def guess_to_state(self):
        """ 
        State is as follows:
        First position is number of guesses left
        Next 26 positions is whether letter has been attempted or not
        For positions 27-417, it is as follows:
        A: [No, Maybe, Yes] for first letter
        B: [No, Maybe, Yes] for first letter
        ...
        Until you go through all 5 letters of word.
        """
        if self.num_guesses == 0:
            self.state[0] = 1
            # Mark every letter in every position as maybe
            for i in range(130):
                self.state[58 + 1 + 3 * i] = 1
        elif self.num_guesses > 0:
            self.state[:6] = np.zeros(6)
            self.state[self.num_guesses] = 1
            guess = self.wordle.guesses[-1]
            hint = self.wordle.hints[-1]

            # No pass through
            for i in range(5):
                val = ord(guess[i]) - ord('a')
                if hint[i] == 0:
                    for j in range(5):
                        self.state[58 + 26 * 3 * j + 3 * val] = 1
                        self.state[58 + 26 * 3 * j + 3 * val + 1] = 0
                        self.state[58 + 26 * 3 * j + 3 * val + 2] = 0
                if hint[i] == 1:
                    self.state[32 + val] = 1
                    self.state[58 + 26 * 3 * i + 3 * val] = 1
                    self.state[58 + 26 * 3 * i + 3 * val + 1] = 0
            # Yes Pass Through
            for i in range(5):
                val = ord(guess[i]) - ord('a')
                self.state[6 + val] = 1
                # If green for a certain letter, make sure all other letters cannot be in that place.
                if hint[i] == 2:
                    self.state[32 + val] = 1
                    for j in range(26):
                        self.state[58 + 26 * 3 * i + 3 * j + 1] = 0
                        self.state[58 + 26 * 3 * i + 3 * j] = 1
                    self.state[58 + 26 * 3 * i + 3 * val + 2] = 1
                    self.state[58 + 26 * 3 * i + 3 * val] = 0

    def step(self, action):
        self.num_guesses += 1
        __ = self.wordle.guess(self.total_words[action])
        self.guess_to_state()
        reward = 0
        if self.num_guesses == 1 and params['info_reward']:
            info = 5*self.bits_info[action][0] / np.max(self.bits_info)
            reward += info
        if self.wordle.win:
            # Reward more for faster wins
            reward += self.reward_win*(7 - self.num_guesses)
        elif not self.wordle.win and self.wordle.over:
            reward += self.reward_lose
        self.rewards.append(reward)
        state = self.state
        over = self.wordle.over
        return state, reward, over
