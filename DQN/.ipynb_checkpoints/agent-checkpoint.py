from collections import deque
import random
import numpy as np
from model import mlp

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95 #Discount_rate
        self.epsilon = 1.0 #Exploraion rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
                            
    def act(self, state):
        #Epsilon Greedy
        if np.random.rand() <= self.epsilon:
#             print('Random action...')
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
#         print('Predicted action...')
        return np.argmax(act_values[0]) #Return action with the maximum action values
    
    def replay(self, batch_size = 32):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])
        
        #Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis = 1)
        print('Target:\t', target)
        #End state target is reward itself (no lookahead)
        target[done] = rewards[done]
        
        #Q(s, a)
        target_f = self.model.predict(states)
        print('Target F:\t', target_f)
        #make the agent to approximately map the current state to future discounted reward
        print('>>>\t',target_f[range(batch_size), actions])
        target_f[range(batch_size), actions] = target
        print('>>>\t',target_f[range(batch_size), actions])
        print(xxx)
        
        self.model.fit(states, target_f, epochs= 1,  verbose= 0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)