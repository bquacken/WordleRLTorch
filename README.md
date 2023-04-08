# Reinforcement Learning with Wordle

This project was inspired by [3Blue1Brown's](https://www.youtube.com/watch?v=v68zYyaEmEA) video on Wordle Strategies. After implementing the greedy one-step strategy presented in the video, I wanted to see how an RL agent could do in comparison. The RL implementation of the Advantage Actor Critic was heavily influenced by [Andrew Ho](https://wandb.ai/andrewkho/wordle-solver/reports/Solving-Wordle-with-Reinforcement-Learning--VmlldzoxNTUzOTc4) with some modifications. Below we can see an animation of how the Transformer Actor-Critic model performs:



<center>    

https://user-images.githubusercontent.com/68025393/230702975-829ee72c-5f94-46e0-ad49-507a4b97638a.mov

</center>​

# Models
Two different neural network architectures were tested. First, a standard MLP consisting of 3 "head" layers with 256 neurons each. This was then fed into two sets of two layers each with 256 neurons: one set of layers outputting the critic value and the second set outputting character-position logits.

The transformer was (of course) heavily inspired by the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper. One caveat is that the multihead attention layers used an outer product of the Keys and Queries, since each were one dimensional. It would be interesting to try a different state which used character embeddings to try something closer to the original transformer architecture.



# State and Action Representation

The State Vector of length 448 is as follows:

- Position 0-5 is the number of guesses remaining in the game as a one-hot vector
- Positions 6-31 document which letters have been attempted before, being a 1 if they had and a 0 if they have not
- Positions 32-57 document which letters are known to be in the word, corresponding to a 1 if that letter is known to be yellow or green after guessing.
- Positions 58 - 187 correspond to the concatenation of 26 one-hot vectors of length 3 corresponding to [No, Maybe, Yes]. This gives information for each letter of whether it belongs in the first position or not, or whether it is not known.
- Positions 188-447 repeat the idea for positions 58-187 of the state vector, but corresponding to the second letter, third letter, and so on in the word.

The key difference between my implementation of the state and what I was referencing is the addition of positions 27-52, as I felt that there could be more information added for getting a yellow letter. Without positions 27-52, I felt that having a yellow letter only changed the state in the sense of "this letter does not belong in this position," while leaving out the caveat of "but this letter IS found in the word."

The networks output a vector of length 130, where the value of each entry corresponds to the preference of a letter in a certain placement. After outputting this vector, it is multiplied by a matrix of size (total number of words) by 130, in the case of the full game 12,972 by 130. Each row of this matrix looks like 5 one-hot vectors of length 26 concatenated together, displaying which letters showed up in which position. This can then be fed through a softmax layer and sampled to get the action for the given state.


## Comparing Guesses Between Models
<center>

| 3Blue1Brown | MLP | Transformer |
| :-- | :-- | :-- |
| rates | sleet | stare |
| being | mourn | glint | 
|movie | chode | <span style="color:green"> diode </span>  |
| <span style="color:green"> diode </span> | abode | | 
| | <span style="color:green"> diode </span>  | 

</center>


## Comparing Performance Between the Models

<center>

| Model | Wins (out of 2309) | Average Score |
| :-- | :-- | :-- |
| 3B1B | 2309 | 3.87 |
| MLP | 2179 | 3.91 | 
| Transformer | 2298 | 3.78 |

</center>

Note that the above table only considers winning games into the average. Below shows a histogram of the number of turns each strategy took to win the game, with 7 turns being a loss. 
    
<center>    
![hist](https://user-images.githubusercontent.com/68025393/230703014-b66cbdb5-89a0-42d7-b6b9-ae15ea4f91e8.jpeg)

</center>​



​    






