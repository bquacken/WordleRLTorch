import numpy as np
import os
from wordle.get_word_funcs import get_words

'''The following functions are helpful for efficiently converting the hint from wordle into a value in [0, 242] 
using ternary expansion where:
0 = Grey
1 = Yellow
2 = Green'''


def array_to_num(arr: np.ndarray) -> np.int:
    assert len(arr) == 5
    integer = 0
    for i in range(5):
        integer += arr[4 - i] * 3 ** i
    return np.uint8(integer)


def num_to_array(integer: int) -> np.ndarray:
    assert 0 <= integer < 243
    arr = np.zeros(5)
    i = 1
    while integer > 0:
        arr[5 - i] = integer % 3
        integer = integer // 3
        i += 1
    return arr


def similarity_value(guess: str, answer: str) -> np.ndarray:
    """
    Generate the hint array of greens, yellows, and greys.
    A green pass followed by a yellow pass was determined to be necessary due to the complications with a guess/answer
    having letter(s) show up more than once. For example if guess = speed and answer = scope then the hint should be
    [2, 1, 1, 0, 0] and not [2, 1, 1, 1, 0]
    """
    w1 = np.array([ord(c) for c in guess])
    w2 = np.array([ord(c) for c in answer])
    assert len(w1) == len(w2)
    sim_arr = np.zeros(len(w1))
    # Check for Greens
    for k in range(len(w1)):
        if w1[k] == w2[k]:
            sim_arr[k] = 2
            w2[k] = 0
    # Check for Yellow
    for k in range(len(w1)):
        if w2[k] == 0:  # Already been dealt with in previous for loop
            continue
        if w1[k] in w2:
            sim_arr[k] = 1
            loc = np.where(w2 == w1[k])[0]
            w2[loc[0]] = 1  # Replace character to deal with duplicate letter issue
    return sim_arr


def compute_similarity_matrix() -> np.ndarray:
    """
    This computation takes a while! Using 3b1b strategy, it is easiest to have the similarity_value precomputed so
    that the strategy simply has to lookup information rather than compute it. This function returns a matrix where
    entry (i,j) returns the similarity_value expressed as a ternary expansion where word[i]
    is the guess and word[j] is the answer
    """
    total_words = get_words()
    len1 = len(total_words)
    array = np.zeros((len1, len1), dtype=np.uint8)
    for i in range(len1):
        w1 = total_words[i]
        for j in range(len1):
            w2 = total_words[j]
            val = np.uint8(array_to_num(similarity_value(w1, w2)))
            array[i, j] = val

    return array


def get_similarity_matrix() -> np.ndarray:
    if not os.path.exists('wordle/data/similarity_matrix.npy'):
        print('Creating Similarity Matrix, this may take a while...')
        sim_mat = compute_similarity_matrix()
        np.save('wordle/data/similarity_matrix', sim_mat)
    else:
        sim_mat = np.load('wordle/data/similarity_matrix.npy')
    return sim_mat
