import numpy as np
import time
import pyprind
import sys


class SARSA():
    '''
    Implementation of Tabular SARSA
    '''
    def __init__(self, env, agent, num_episodes, num_steps, eps, lr, disc, seed):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.eps = eps
        self.lr = lr
        self.disc = disc
        self.seed = seed

    def choose_action(self, s):
        '''
        Epsilon Greedy Action Selection
        
        Returns : next action
        '''
        if np.random.rand() > self.eps:
            return self.agent.act(s)
        else:
            return np.random.randint(0, 4)

    def run(self):
        '''
        Implementation of the SARSA Training Loop
        
        Returns : episode_rewards : sequence of episode rewards
                  all_trajectories : list of all the trajectories
                                     taken by the agent during training
                  training_time : float (in seconds)
                  steps_takens : dictionary - key=episode, value=number of steps takes
                  
        '''
        np.random.seed(self.seed)
        episode_rewards = []
        steps_taken = {}
        all_trajectories = {}
        start = self.agent.state

        bar = pyprind.ProgBar(self.num_episodes, stream=sys.stdout,
                              track_time=False)
        start_time = time.time()

        #SARSA Training Loop
        for e in range(self.num_episodes):
            s = start
            a = self.choose_action(s) 
            total_reward = 0
            total_steps = 0
            traj = []

            for step in range(self.num_steps):
                traj.append(s)
                
                r, s_1 = self.env.step(s, a)
                total_reward += r
                a_1 = self.choose_action(s_1)  
                
                #SARSA update function, on-policy
                self.agent.q_table[s][a] += self.lr * (
                    r + self.disc * self.agent.q_table[s_1][a_1] - self.agent.q_table[s][a]
                )

                #end if the agent reached the target
                if r == 100:
                    traj.append(s_1)
                    steps_taken[e] = len(traj)   
                    break
                s = s_1
                a = a_1
                
            #store data for logging/plots
            episode_rewards.append(total_reward)
            all_trajectories[e] = traj
            bar.update()

        end_time = time.time()
        return episode_rewards, all_trajectories, end_time-start_time, steps_taken

