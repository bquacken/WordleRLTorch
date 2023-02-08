import numpy as np
import os

from wordle.word_functions import similarity_value, get_similarity_matrix
from wordle.get_word_funcs import get_words, get_freqencies
'''
    3b1b implementation of Expected Information. When given a guess = word[i], this function searches the ith row
    of the similarity matrix and find the probabilities that guess yields hint[j] for all j
'''


def safe_log(p):
    if p == 0:
        return 0
    else:
        return np.log2(p)


def expected_information(guess, sim, word_list):
    totals = np.zeros(243)
    index = np.where(word_list == guess)[0][0]
    for i in range(len(sim)):
        totals[sim[index, i]] += 1
    totals = totals / len(word_list)
    result = sum([-p * safe_log(p) for p in totals])
    return result


class Player:
    def __init__(self):
        self.n_letters = 5
        self.n_guesses = 6
        self.total_words, self.answer_words, self.acceptable_words = get_words(True)
        self.similarity_matrix = get_similarity_matrix()
        self.word_freq = get_freqencies()
        self.num_guesses = 0
        self.guesses = []
        self.hints = []

    def reset(self):
        self.num_guesses = 0
        self.guesses = []
        self.hints = []
        self.total_words, self.answer_words, self.acceptable_words = get_words(True)
        self.similarity_matrix = get_similarity_matrix()
        self.word_freq = get_freqencies()

    def guess(self, guess, hint):
        self.num_guesses += 1
        self.guesses.append(guess)
        self.hints.append(hint)
        self.update()

    def update(self):
        """
        Whether to only consider words that agree with all previous hints.
        As of now, I have not done anything for a non-greedy strategy.
        """
        hint = self.hints[-1]
        guess = self.guesses[-1]
        arr = []
        for k in range(len(self.total_words)):
            word = self.total_words[k]
            temp_hint = list(similarity_value(guess, word))
            if (hint == temp_hint).all():
                arr.append(k)
        # print(arr)
        if len(arr) == 0:
            raise Exception('Word list is empty!')
        self.total_words = self.total_words[arr]
        self.word_freq = self.word_freq[arr]
        temp1 = []
        temp2 = []
        for word in self.total_words:
            if word in self.answer_words:
                temp1.append(word)
            else:
                temp2.append(word)
        self.answer_words = np.array(temp1)
        self.acceptable_words = np.array(temp2)
        self.similarity_matrix = self.similarity_matrix[arr, :]
        self.similarity_matrix = self.similarity_matrix[:, arr]

    def exp_info_strategy(self):
        if self.num_guesses == 0 and os.path.exists('wordle/data/InitialStrategy.npy'):
            info = np.load('wordle/data/InitialStrategy.npy')
            return self.total_words[np.argmax(info)]
        elif self.num_guesses == 0 and not os.path.exists('wordle/data/InitialStrategy.npy'):
            info = np.array([freq * expected_information(word, self.similarity_matrix,
                                                         self.total_words) for freq, word
                             in zip(self.word_freq, self.total_words)])
            np.save('wordle/data/InitialStrategy.npy', info)
            return self.total_words[np.argmax(info)]
        info = np.array([freq * expected_information(word, self.similarity_matrix,
                                                     self.total_words) for freq, word
                         in zip(self.word_freq, self.total_words)])
        return self.total_words[np.argmax(info)]
