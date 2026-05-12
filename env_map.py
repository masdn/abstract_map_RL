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

        #change target cell value 
        self.grid[target[0]][target[1]] = 0.5 
        

    def s1(self, state, action):
        '''
        naive reward strategy
        returns next_state and reward
        '''
        x_a, y_a = action
        next_state = None
        reward = None

        #reward conditionals based on value at (x_a, y_a)
        if x_a == self.target[0] and y_a == self.target[1]: #target reached
            next_state = (x_a, y_a)
            reward = 100
        elif self.grid[x_a][y_a] == 1:                      #blank space /step
            next_state = (x_a, y_a)
            reward = -2
        elif self.grid[x_a][y_a] == 0:                      #obstacle
            next_state = state
            reward = -10
        else:
            print('Error: See env_map.py s1()')

        return next_state, reward
        

    def s2(self, state, action):
        '''
        Reward strategy with Euclidean distance.
        
        Applies a weighted negative reward that 
        calculates the Euclidean distance between
        (x_a,y_a) and the target.
        
        returns next_state and reward
        '''
        x_a, y_a = action
        
        dt_x = abs(self.target[0] - x_a)
        dt_y = abs(self.target[1] - y_a)
        
        t_distance = ((dt_x**2)+(dt_y**2)) ** 0.5
        
        next_state = None
        reward = None

        #reward conditionals based on value at (x_a, y_a)
        if x_a == self.target[0] and y_a == self.target[1]: #target reached
            next_state = (x_a, y_a)
            reward = 100
        elif self.grid[x_a][y_a] == 1:                      #blank space
            next_state = (x_a, y_a) 
            reward = -0.05*t_distance                   #euclidean distance reward factor
        elif self.grid[x_a][y_a] == 0:                      #obstacle
            next_state = state
            reward = -10
        
        else:
            print('Error: See env_map.py s1()')

        return next_state, reward

    
    def step(self, state, action):
        '''
        Params : state : (x,y) - position
                 action : int - between 0-3
        action: int  0=UP 1=DOWN 2=LEFT 3=RIGHT
        returns: (next_state, reward)
        '''
        x, y = state

        #moves the x or y coordinate based on the action passed
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        x_dir, y_dir = moves[action]
        next_pos = (x + x_dir, y + y_dir)

        if self.type == 's1':
            next_state, reward = self.s1(state, next_pos)
        else:
            next_state, reward = self.s2(state, next_pos)

        self.state = next_state
        return reward, next_state

    def value_at(self, pos):
        '''
        returns the value at position (x,y)
        in the environment 
        '''
        return self.grid[pos[0]][pos[1]]

