from env_map import env
from agent import agent

EPSILONs = []
GAMMAs = []


# init env
mp = 'map3'
#mp = utils.abstract_map(mp)
n = 528 #test values
m = 532
target_pos = (n,m)
env_s1 = env(mp,target_pos, 'S1') # S1 reward strat
env_s2 = env(mp,target_pos, 'S2') # S2 reward strat
agent =  agent()

'''
Complexity of the Map
'''
'''
Exploration Rate
'''
'''
Discount Value
'''
'''
Reward Stratedy 
'''
# run sarsa 
sarsa_res1 = sarsa(agent, env_s1)
sarsa_res2 = sarsa(agent, env_s2)
# run q-learning
q_learn_res1 = q_learning(agent, env_s1)
q_learn_res2 = q_learning(agent, env_s2)

#print save all tables 


