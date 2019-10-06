# Importing Dependencies
import os
import sys
import time
import numpy as np
import pandas as pd
import gym
from pyglet.window import key
from gym.envs.box2d.car_racing import *


# To log key presses
def key_press(k, mod):
    global end
    global restart
    global action
    
    if k==key.END:
        end = True
    if k==key.RETURN:
        restart = True
    if k==key.LEFT:
        action[0] = -1.0
    if k==key.RIGHT:
        action[0] = +1.0
    if k==key.UP:
        action[1] = +1.0
    if k==key.DOWN:
        action[2] = +0.8   
 

def key_release(k, mod):
    global action
    if k==key.LEFT  and action[0]==-1.0:
        action[0] = 0
    if k==key.RIGHT and action[0]==+1.0:
        action[0] = 0
    if k==key.UP:
        action[1] = 0
    if k==key.DOWN:
        action[2] = 0

end = False
restart = False
env = CarRacing()
env.render()
action = np.array([0, 0, 0], dtype=np.float16)
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release


# Function for generating data by playing
def start_playing(num_samples):
    global restart
    global action
    global end
    
    l = 0
    r = 0
    u = 0
    d = 0
    nk = 0
    
    left = np.array([-1, 0, 0], dtype=np.float16)
    right = np.array([1, 0, 0], dtype=np.float16)
    up = np.array([0, 1, 0], dtype=np.float16)
    down = np.array([0, 0, 0.8], dtype=np.float16)
    no_key = np.array([0, 0, 0], dtype=np.float16)
    
    left_processed = np.array([1, 0, 0, 0, 0], dtype=np.float16)
    right_processed = np.array([0, 1, 0, 0, 0], dtype=np.float16)
    up_processed = np.array([0, 0, 1, 0, 0], dtype=np.float16)
    down_processed = np.array([0, 0, 0, 1, 0], dtype=np.float16)
    no_key_processed = np.array([0, 0, 0, 0, 1], dtype=np.float16)
    
    x = np.empty((num_samples * 5, 96, 96, 3), dtype=np.float16)
    y = np.empty((num_samples * 5, 5), dtype=np.float16)
    frame = 0
    max_frames = num_samples * 5
    num_episode = 1
    
    while True:
        prev_observation = env.reset()
        restart = False
        
        while True:
            # Render the env
            env.render()
            
            # Log the key and frame
            if np.array_equal(action, left) and l<num_samples:
                l+=1
                x[frame] = prev_observation
                y[frame] = left_processed
                frame+=1
            elif np.array_equal(action, right) and r<num_samples:
                r+=1
                x[frame] = prev_observation
                y[frame] = right_processed
                frame+=1
            elif np.array_equal(action, up) and u<num_samples:
                u+=1
                x[frame] = prev_observation
                y[frame] = up_processed
                frame+=1
            elif np.array_equal(action, down) and d<num_samples:
                d+=1
                x[frame] = prev_observation
                y[frame] = down_processed
                frame+=1
            elif np.array_equal(action, no_key) and nk<num_samples:
                nk+=1
                x[frame] = prev_observation
                y[frame] = no_key_processed
                frame+=1
            else:
                # Ignore the frame
                pass
            
            observation, reward, done, info = env.step(action)
            prev_observation=observation
            if(done or restart):
                break
            
        # Episode end
        if frame >= max_frames or end:
            print("Collected Samples...")
            print(f"Left: {l} Right: {r}, Up: {u}, Down: {d}, No_Key: {nk}")
            env.close()
            break
        else:
            print(f"Episode: {num_episode}:: Collected Samples...")
            print(f"Left: {l} Right: {r}, Up: {u}, Down: {d}, No_Key: {nk}")
            num_episode+=1
            time.sleep(2)
            
    # max_frames complete
    np.save('../Data/frames.npy', x)
    Y = pd.DataFrame(y)
    Y.columns = ['left', 'right', 'up', 'down', 'no_key']
    Y.to_csv('../Data/controls.csv') 


if __name__ == '__main__':
    if len(sys.argv)<=1:
        raise ValueError('Please input num_samples..')
    else:
        num_samples = int(sys.argv[1])
        print(f"Starting environment for collecting {num_samples} samples..")
        start_playing(num_samples)
        print("Done..")
