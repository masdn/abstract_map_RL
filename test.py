import numpy as np

class policy_evaluator():
    '''
    Tests the final policies learned
    by an agent and returns the results
    '''
    def __init__(self, policy, env, num_steps, seed):
        self.policy = policy #agent object with a fully-trained policy
        self.env = env
        self.num_steps = num_steps
        self.seed = seed

    def eval(self):
        '''
        Loops over all possible starting points and 
        asks the policy what to do. Tracks all valid
        paths.
        
        Returns test_accuracy
        '''
        np.random.seed(self.seed)

        #get list of all blank squares (i.e valid starting states)
        start_states = self.get_states()
        num_valids = 0
        steps_taken = {}

        #Test loop over all position starting states
        for e, start in enumerate(start_states):
            target_hit = False
            obstacle_hit = False
            s = start

            for step in range(self.num_steps):
                a = self.policy.act(s)
                r, s_1 = self.env.step(s, a)
                
                if r == -10:                       #obstacle hit
                    obstacle_hit = True
                if r == 100:                       #target reached
                    target_hit = True
                    break

                s = s_1
            
            if target_hit and not obstacle_hit:
                num_valids += 1
        
        #numbers of valid/perfect paths over the number of start_states test on
        test_accuracy = num_valids / len(start_states) if start_states else 0.0

        return test_accuracy
        
    def get_states(self):
        '''
        loops over environment cells and returns the all valid positions
        i.e not a target cell and not an obstacle cell.
        '''
        start_states = []
        for x in range(self.env.grid.shape[0]):
            for y in range(self.env.grid.shape[1]):
                if self.env.value_at((x, y)) == 1 and (x, y) != self.env.target:
                    start_states.append((x, y))
        return start_states
