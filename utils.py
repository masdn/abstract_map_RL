import numpy as np
import pygame
from skimage.measure import block_reduce
import os
import pandas as pd
import matplotlib.pyplot as plt

def log(results, file_path):
    '''
    Params: results : dictionary
            file_path : string
            
    Makes a pandas DataFrame and extract results to a .csv
    '''
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(results, dict):
        results = [results]
    df = pd.concat([pd.DataFrame(r) for r in results if r is not None], ignore_index=True)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    



def animate(map_,w,h,start, target, traj, title=''):
    '''
    Params: map_ : 2D array of a greyscale map
            w : width
            h : height
            start : (x,y)  initial agent position
            target : (x,y) position of target
            traj : [(x_0,y_0),...,(x_n,y_n)]
                      list of traj taken
    Animates an agent's trajectory using pygame library.
    loops over all traj (x_i,y_i) and calls draw()
    '''
    pygame.init()
    cell_size = 20
    pygame.display.set_caption(title)
    window = pygame.display.set_mode((w*cell_size,h*cell_size))
    
    draw(map_, w, h, window, cell_size, target,start)           #draw starting state

    #draw each action that was taken in the trajectory
    for (x,y) in traj:
        draw(map_, w, h, window, cell_size, target,(x,y))
        pygame.time.delay(100)                                  #delay by 0.1s


    pygame.quit()

def draw(map_,w,h, window, cell_size, target, agent_pos):
    '''
    Draws an individual frame from the trajectory
    '''
    #put all cells on the window
    for x in range(w):
        for y in range(h):
            if map_[x][y] == 0:
                pygame.draw.rect(window, 'black', 
                             (x*cell_size , y*cell_size , cell_size, cell_size))
            elif map_[x][y] == 1:
                pygame.draw.rect(window, 'white', 
                             (x*cell_size , y*cell_size , cell_size, cell_size))
    #draw target
    pygame.draw.rect(window, 'green', 
                    (target[0]*cell_size , target[1]*cell_size , cell_size, cell_size))
    #draw agent
    pygame.draw.rect(window, 'orange', 
                    (agent_pos[0]*cell_size , agent_pos[1]*cell_size , cell_size, cell_size))
    pygame.display.flip()



def plot_rewards(num_episodes, episode_rewards, save_path, title=''):
    '''
    Simple plot and save using matplotlib
    '''
    
    episodes = list(range(1, num_episodes + 1))
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def bmp_to_mat(abs_siz, fname):
    '''
    Params: abs_siz : int - abstraction size
            fname : String - name of file (map name)

    Loads the .bmp files into a pygame surface, then to a matrix.
    Next, it utilizes skimage.measure.block_reduce and np.min function 
    to downsample the greyscale matrix to the abs_siz.
    Finally, an additional row and col of zeros is added which will 
    make the bounds just be another obstacle around the whole
    map.

    Return both the pygame surface and the fully abstracted map
    '''
    surface = pygame.image.load(f'bmps/{fname}.bmp')
    arr = pygame.surfarray.array3d(surface)
    arr = arr[:, :, 0]                                 #get only one of RGB
    greyscale_arr = (arr > 0).astype(int)              #sets all blank squares to 1
    
    pooled_map = block_reduce(greyscale_arr, abs_siz, np.min)

    #add "obstacle" border
    pooled_map = np.vstack([np.zeros((1, pooled_map.shape[1])), pooled_map])
    pooled_map = np.vstack([pooled_map, np.zeros((1, pooled_map.shape[1]))])
    pooled_map = np.hstack([np.zeros((pooled_map.shape[0], 1)), pooled_map])
    pooled_map = np.hstack([pooled_map, np.zeros((pooled_map.shape[0], 1))])

    return surface, pooled_map



def get_results_dict(FNAMEs=[]):
    '''
    Params: FNAMEs : list of map file names
    returns dictionary templates to help
    standarize the experiment results 
    across all experiments
    '''
    dict = {}
         
    per_map = {
            'timestamp' : [],
            'experiment': [],
            'num_episodes' : [],
            'num_steps' : [],
            'method': [],
            'reward_strat': [],
            'lr': [],
            'explore_rate': [],
            'disc_rate': [],
            'time_cost': [],
            'shortest_path': [],
            'test_accuracy': [],
            
    }
    dict = {
        #make a sub-dictionary for each map
                FNAMEs[0] : {k: [] for k in per_map},
                FNAMEs[1] : {k: [] for k in per_map},
                FNAMEs[2] : {k: [] for k in per_map},
                FNAMEs[3] : {k: [] for k in per_map}
    }
    return dict



