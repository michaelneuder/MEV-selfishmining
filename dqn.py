import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import copy

EPISODES = 1000
TERMINATE_BLOCK = 5
WHALE_REWARD = 10

class Environment:
    def __init__(self, num_players): 
        self.num_players = num_players

        # The agent is able to mine on any of the forks.
        self.num_actions = self.num_players

        # State is a vector of fork lengths and a matrix of blocks owned.
        self.state_size = self.num_players + self.num_players ** 2 

    def reset(self):
        # Number of steps in the current episode.
        self.num_steps = 0

        # The length of each miner fork.
        self.fork_lengths = np.zeros(self.num_players)

        # The number of blocks owned by each miner in each fork.
        self.miner_blocks = np.zeros((self.num_players, self.num_players))

        # Initialize a random miner to mine the first block. 
        init_miner = np.random.choice(self.num_players)
        
        self.fork_lengths[init_miner] = 1
        self.miner_blocks[init_miner, init_miner] = 1

        return self.getState(self.fork_lengths, self.miner_blocks)

    def getState(self, fork_lengths, miner_blocks):
        return np.concatenate((fork_lengths, miner_blocks.flatten()))

    '''
    This is the most complex part of the code. In order to figure out what
    every agents actions are going to be, we need to shuffle the indeces.
    We have to first swap elements of the fork lengths. For example if the
    for lengths are [1, 2, 3], and we pass ind=1, we are switching the first
    two elements, so the output fork lengths are [2, 1, 3].  For the miner
    blocks, we need to switch rows and switch elements withing those rows.
    For example again with ind=1, if the miner blocks are
    [[1,2,3]
    [4,5,6]
    [7,8,9]]
    then the output should be
    [[5,4,6]
    [2,1,3]
    [7,8,9]]
    '''
    def permuteState(self, ind):
        # Swap elements.
        temp_fork_lengths = copy.deepcopy(self.fork_lengths)
        temp_fork_lengths[0], temp_fork_lengths[ind] =\
            temp_fork_lengths[ind], temp_fork_lengths[0]

        # Swap rows.
        temp_miner_blocks = copy.deepcopy(self.miner_blocks)
        temp_miner_blocks[[0, ind]] = self.miner_blocks[[ind, 0]]
        
        # Swap elements of swapped rows.
        temp_miner_blocks[0][0], temp_miner_blocks[0][ind] =\
            temp_miner_blocks[0][ind], temp_miner_blocks[0][0]
        temp_miner_blocks[ind][0], temp_miner_blocks[ind][ind] =\
            temp_miner_blocks[ind][ind], temp_miner_blocks[ind][0]

        return temp_fork_lengths, temp_miner_blocks

    '''
    Makes a step in the environment based on the current agent network. Returns
    the new state, the reward, if the episode is completed, and the action that
    the 0th agent took.
    '''
    def step(self, agent):
        self.num_steps += 1

        agent_actions = []
        # Loop through each miner to determine what action they will take.
        for i in range(self.num_players):
            permute_forks, permute_blocks = self.permuteState(i)
            action = agent.act(self.getState(permute_forks, permute_blocks))
            agent_actions.append((i, action))
        
        # Random miner given the next block. 
        miner_fork = agent_actions[
            np.random.choice(np.arange(len(agent_actions)))]
        
        # Increment fork length and miner blocks
        self.fork_lengths[miner_fork[1]] += 1
        self.miner_blocks[miner_fork[0]][miner_fork[1]] += 1

        new_state = self.getState(self.fork_lengths, self.miner_blocks)

        # Terminate when someone reaches five blocks.
        done = False
        reward = 0
        if TERMINATE_BLOCK in self.fork_lengths:
            done = True
            # Agent wins! Earn whale rewards.
            if self.fork_lengths[0] == TERMINATE_BLOCK:
                reward = WHALE_REWARD - self.num_steps + self.miner_blocks[0,0]
            # Agent loses.
            else:
                win_ind = np.argwhere(self.fork_lengths == TERMINATE_BLOCK)[0,0]
                reward = - self.num_steps + self.miner_blocks[0, win_ind]
        
        return new_state, reward, done, agent_actions[0][1]        

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = Environment(num_players=3)
    env.reset()

    # Initialize DQN agent.
    state_size = env.state_size
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)
    
    done = False
    batch_size = 32

    for e in range(1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            next_state, reward, done, action = env.step(agent)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, time: {}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        