import numpy as np
from collections import defaultdict

class agent():
    def __init__(self, env, starting_pos):
        self.env = env
        self.state = starting_pos
        # Q[state][action]: action is int 0=UP 1=DOWN 2=LEFT 3=RIGHT
        self.q_table = defaultdict(lambda: {a: 0.0 for a in range(self.env.total_actions)})

    def act(self, state):
        '''
        Greedy: return action with highest q_value for current state.
        '''
        return max(self.q_table[state], key=self.q_table[state].get)
