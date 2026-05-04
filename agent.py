import numpy as np

class agent():
    def __init__(self, starting_pos):
        self.pos = starting_pos
        self.q_table = []
    def act(self, pos):
        '''
        actions = [UP,DOWN,LEFT,RIGHT]
        '''
        x = pos[0]
        y = pos[1]
        neighbors = [self.q_table[x-1][0], #up
                     self.q_table[x+1][0], #down
                     self.q_table[0][y-1], #left
                     self.q_table[0][y+1]] #right        

        return np.argmax(neighbors)
