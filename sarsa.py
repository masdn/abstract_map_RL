import numpy as np
import time


class SARSA():
    def __init__(self, env, agent, num_episodes, num_steps, eps, lr, disc):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.eps = eps
        self.lr = lr
        self.disc = disc

    def choose_action(self, s):
        if np.random.rand() > self.eps:
            return self.agent.act(s)
        else:
            return np.random.randint(0, 4)

    def run(self):
        episode_rewards = []
        steps_taken = {}
        all_trajectories = {}
        start = self.agent.state

        start_time = time.time()
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
                
                self.agent.q_table[s][a] += self.lr * (
                    r + self.disc * self.agent.q_table[s_1][a_1] - self.agent.q_table[s][a]
                )

                if r == 100:
                    traj.append(s_1)
                    steps_taken[e] = step   #store 
                    break
                s = s_1
                a = a_1
                

            episode_rewards.append(total_reward)
            all_trajectories[e] = traj

        end_time = time.time()
        return episode_rewards, all_trajectories, end_time-start_time, steps_taken

