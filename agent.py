import numpy as np

class agent():
    def __init__(self, starting_pos):
        self.state = starting_pos
        self.state = None
        self.q_table = {}

    def act(self, pos):
        '''
        actions = [UP,DOWN,LEFT,RIGHT]
        '''
        x = pos[0]
        y = pos[1]
        s_t = (x, y)
        a_t = ()
        r_t1 = 0
        if s_t not in self.q_table:
            self.q_table[s_t][a_t] = r_t1

        neighbors = [self.q_table[s_t][(x-1, 0)], #up
                     self.q_table[s_t][(x+1, 0)], #down
                     self.q_table[s_t][(0, y-1)], #left
                     self.q_table[s_t][(0, y+1)]] #right        

        return np.argmax(neighbors)

    def update_q(self, next_state, reward, lr, disc):
        '''
        applies the Q(s,a) update function
        '''
        #update q_table
        self.state = next_state
        q_sa = self.q_table[next_state][reward] 
        q_sa_1 = self.q_table[next_state][reward] + reward #TODO i know this wrong btw
        self.q_table[next_state] = q_sa + lr(reward + disc*q_sa_1 - q_sa)
