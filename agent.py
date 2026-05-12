import numpy as np
from collections import defaultdict

class agent():
    def __init__(self, env, start_state):
        self.env = env
        self.state = start_state
        #set all q-values to 0.0 for all 4 actions
        self.q_table = defaultdict(lambda: {a: 0.0 for a in range(self.env.total_actions)})

    def act(self, state):
        '''
        return action with highest q_value for current state.
        '''
        return max(self.q_table[state], key=self.q_table[state].get)
