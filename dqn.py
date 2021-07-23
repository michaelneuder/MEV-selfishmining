import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import copy
import pandas as pd

EPISODES = 100
TERMINATE_BLOCK = 2
WHALE_REWARD = 10

class Environment:
    def __init__(self, num_players): 
        self.num_players = num_players

        # The agent is able to mine on any of the forks.
        self.num_actions = self.num_players

        # State is a vector of fork lengths and a vector of blocks owned per 
        # fork.
        self.state_size = 2 * self.num_players 

    def reset(self):
        # Number of steps in the current episode.
        self.num_steps = 1

        # The number of blocks owned by each miner in each fork.
        self.miner_blocks = np.zeros((self.num_players, self.num_players))

        # Initialize a random miner to mine the first block. 
        init_miner = np.random.choice(self.num_players)
        self.miner_blocks[init_miner, init_miner] = 1
        # self.miner_blocks[0, 0] = 1

        return self.getState(self.miner_blocks)

    '''
    Returns the state which is a vector of lengths of each fork and a vector 
    of the number of blocks owned by the 0th agent in each fork. The state is
    size (2 * self.num_agents).
    '''
    def getState(self, miner_blocks):
        fork_lengths = np.sum(miner_blocks, axis=0)
        agent_blocks = miner_blocks[0]
        return np.reshape(np.concatenate((fork_lengths, agent_blocks)),
                          (1, self.state_size))


    '''
    This is the most complex part of the code. In order to figure out what
    every agents actions are going to be, we need to shuffle the indices.
    We need to switch rows and switch elements within those rows.
    For example again with ind=1, if the miner blocks are
    [[1,2,3]
    [4,5,6]
    [7,8,9]]
    then the output should be
    [[5,4,6]
    [2,1,3]
    [7,8,9]]
    '''
    def permuteMinerBlocks(self, ind):
        # Swap rows.
        temp_miner_blocks = copy.deepcopy(self.miner_blocks)
        temp_miner_blocks[[0, ind]] = self.miner_blocks[[ind, 0]]
        
        # Swap elements of swapped rows.
        temp_miner_blocks[0, 0], temp_miner_blocks[0, ind] =\
            temp_miner_blocks[0, ind], temp_miner_blocks[0, 0]
        temp_miner_blocks[ind, 0], temp_miner_blocks[ind, ind] =\
            temp_miner_blocks[ind, ind], temp_miner_blocks[ind, 0]

        return temp_miner_blocks

    '''
    Permutes the action to be in the perspective of the index of the agent.
    '''
    def permuteAction(self, ind, action):
        if action == 0:
            return ind
        elif action == ind:
            return 0
        return action

    '''
    Makes a step in the environment based on the current agent network. Returns
    the new state, the reward, if the episode is completed, and the action that
    the 0th agent took.
    '''
    def step(self, agent):
        self.num_steps += 1

        agent_actions = []
        # Loop through each miner to determine what action they will take.
        for agent_ind in range(self.num_players):
            permute_blocks = self.permuteMinerBlocks(agent_ind)
            action = self.permuteAction(
                agent_ind, agent.act(self.getState(permute_blocks)))
            agent_actions.append((agent_ind, action))
        
        # Random miner given the next block. 
        miner_fork = agent_actions[
            np.random.choice(np.arange(len(agent_actions)))]

        print("    mined block={}".format(miner_fork))
        
        # Increment miner blocks. 
        self.miner_blocks[miner_fork] += 1

        new_state = self.getState(self.miner_blocks)

        # Terminate when any fork reaches TERMINATE_BLOCK length.
        done = False
        reward = 0
        if TERMINATE_BLOCK in np.sum(self.miner_blocks, axis=0):
            done = True
            # Agent wins! Earn whale rewards.
            if np.sum(self.miner_blocks[:,0]) == TERMINATE_BLOCK:
                reward = WHALE_REWARD - self.num_steps + self.miner_blocks[0,0]
                # print("miner wins, steps={}, blocks={}".format(self.num_steps,self.miner_blocks[0,0]))
            # Agent loses.
            else:
                win_ind = np.argwhere(
                    np.sum(self.miner_blocks, axis=0) == TERMINATE_BLOCK)[0,0]
                reward = - self.num_steps + self.miner_blocks[0, win_ind]
                # print("miner loses, steps={}, blocks={}".format(self.num_steps,self.miner_blocks[0,win_ind]))
        
        return new_state, reward, done, agent_actions[0][1]        

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
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
            # print("state={}, action={}, reward={}, next_state={}, done={}"
            #     .format(state, action, reward, next_state, done))
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
    batch_size = 64

    rewards = []
    for e in range(EPISODES):
        print("episode {}".format(e))
        state = env.reset()
        for time in range(1, 500):
            next_state, reward, done, action = env.step(agent)
            agent.memorize(state, action, reward, next_state, done)
            print("    state={}, action={}, next_state={}, done={}, rew={}".format(
                state, action, next_state, done, reward))
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                print("    episode summary: {}/{}, time: {}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, reward, agent.epsilon))
                rewards.append(reward)
                break
            
    
    np.savetxt("rewards.txt", np.asarray(rewards))
    agent.save('trained.model')
        