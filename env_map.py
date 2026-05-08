import numpy as np
import matplotlib.pyplot as plt

class env():
    total_actions = 4  # 0=UP 1=DOWN 2=LEFT 3=RIGHT

    def __init__(self, grid, dims, type_, start_state,target=(0,0)):
        self.grid = grid
        self.dims = dims
        self.type = type_
        self.target = target
        self.state = start_state

        #highlight target cell
        #self.grid[target[0]][target[1]] = 0.3
        #self.grid[self.state[0]][self.state[1]] = 0.8

    def s1(self, state, action):
        x_a = action[0]
        y_a = action[1]
        next_state = None
        reward = None

        if self.grid[x_a][y_a] == 1: #blank space
            next_state = (x_a, y_a)
            reward = -1
        elif self.grid[x_a][y_a] == 0: #obstacle
            next_state = state
            reward = -10
        elif x_a == self.target[0] and y_a == self.target[1]: #target reached
            next_state = (x_a, y_a)
            reward = 100
        else:
            print('Error: See env_map.py s1()')

        return next_state, reward

    def s2(self, state, action):
        x_a = action[0]
        y_a = action[1]
        next_state = None
        reward = None

        if self.grid[x_a][y_a] == 1: #blank space
            next_state = (x_a, y_a) 
            reward = -1 #TODO add manhattan distance here
        elif self.grid[x_a][y_a] == 0: #obstacle
            next_state = state
            reward = -10
        elif x_a == self.target[0] and y_a == self.target[1]: #target reached
            next_state = state
            reward = 100
        else:
            print('Error: See env_map.py s1()')

        return next_state, reward
    def step(self, state, action):
        '''
        action: int  0=UP 1=DOWN 2=LEFT 3=RIGHT
        returns: (next_state, reward)
        '''
        x, y = state
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        x_dir, y_dir = moves[action]
        next_pos = (x + x_dir, y + y_dir)

        if self.type == 's1':
            next_state, reward = self.s1(state, next_pos)
        else:
            next_state, reward = self.s2(state, next_pos)

        self.state = next_state
        return reward, next_state

    def show(self):
        plt.imshow(self.grid, cmap='gray')
        plt.show()

