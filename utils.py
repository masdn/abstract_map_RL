import numpy as np
import pygame
from skimage.measure import block_reduce
import os
import pandas as pd
import matplotlib.pyplot as plt

def animate(map_,w,h,start, target, traj):
    '''
    Params: map_ : 2D array of a greyscale map
            w : width
            h : height
            start : (x,y)  initial agent position
            target : (x,y) position of target
            traj : [(x_0,y_0),...,(x_n,y_n)] 
                      list of traj taken
    loops over all traj (x_i,y_i) and calls draw()
    keeps window open til user exits out
    '''
    cell_size = 20
    window = pygame.display.set_mode((w*cell_size,h*cell_size))
    draw(map_, w, h, window, cell_size, target,start)           #draw starting state
    for (x,y) in traj:
        draw(map_, w, h, window, cell_size, target,(x,y))
        pygame.time.delay(200)                                  #delay by 0.2s

    '''
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:                           #display window til exit
                running = False
    '''
    pygame.quit()

def draw(map_,w,h, window, cell_size, target, agent_pos):
    '''
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



def plot_grid_training_loss(results, save_path, cfg):
    '''
    puts four plots, one for each stock,
    in a 2x2 grid on training loss per epoch
    '''
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    hps = f"hidden={cfg['hidden_size']}  recurrent layers={cfg['num_layers']}  lr={cfg['lr']}  dropout={cfg['dropout']}"
    fig.suptitle(hps, fontsize=10, y=1.01)

    for ax, (stock, epochs, losses) in zip(axes, results):
        ax.plot(epochs, losses, label='train loss')
        ax.set_title(stock)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg MSE Loss')
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    #plt.show()
    plt.close()
    


def bmp_to_mat(abs_siz, fname):
    '''
    gets stock data from yfinance
    '''
    surface = pygame.image.load(f'bmps/{fname}.bmp')
    arr = pygame.surfarray.array3d(surface)
    arr = arr[:, :, 0]
    greyscale_arr = (arr > 0).astype(int)
    
    pooled_map = block_reduce(greyscale_arr, abs_siz, np.min)
    pooled_map = np.vstack([np.zeros((1, pooled_map.shape[1])), pooled_map])
    pooled_map = np.vstack([pooled_map, np.zeros((1, pooled_map.shape[1]))])
    pooled_map = np.hstack([np.zeros((pooled_map.shape[0], 1)), pooled_map])
    pooled_map = np.hstack([pooled_map, np.zeros((pooled_map.shape[0], 1))])

    return surface, pooled_map



def get_results_dict(mode, FNAMEs=[]):
    '''
    returns dictionary templates to help
    standarize the experiment results 
    across all experiments
    '''
    dict = {}
    if mode == 'com':             #Complexity of Map
        per_map = {
            'timestamp' : [],
            'num_episodes' : [],
            'num_steps' : [],
            'method': [],
            'time_cost': [],
            'shortest_path': [],
            'reward_sequence': [],
            'final_q_table': [],
        }
        dict = {
                FNAMEs[0] : {k: [] for k in per_map},
                FNAMEs[1] : {k: [] for k in per_map},
                FNAMEs[2] : {k: [] for k in per_map},
                FNAMEs[3] : {k: [] for k in per_map}
        }
    elif mode == 'eval':
        dict = {
            'stock_name': None,
            'split': None,
            'timestamp': None,
            'rmse': None,
            'mape': None,
            'eval_time': None,
            'input_size': None,
            'hidden_size': None,
            'output_size': None,
            'num_layers': None,
            'dropout': None,
            'lr': None,
            'num_epochs': None,
        }
    return dict


def log(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(results, dict):
        results = [results]
    df = pd.concat([pd.DataFrame(r) for r in results if r is not None], ignore_index=True)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    #print(df.to_string(index=False))


