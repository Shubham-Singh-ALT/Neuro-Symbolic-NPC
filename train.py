import numpy as np
import random
import time
from npc_env import SmartNPCEnv

env = SmartNPCEnv()
q_table = np.zeros((env.grid_size, env.grid_size, len(env.action_space)))

learning_rate = 0.1
discount_factor = 0.9      
exploration_rate = 1.0     
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
num_episodes = 500 

print("🧠 Training the NPC Brain... (Running 500 simulations)")

for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < exploration_rate:
            action = random.choice(env.action_space) 
        else:
            action = np.argmax(q_table[state[0], state[1], :]) 
        new_state, reward, done = env.step(action)
        
        q_table[state[0], state[1], action] = q_table[state[0], state[1], action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_factor * np.max(q_table[new_state[0], new_state[1], :]))
            
        state = new_state
        
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

print("✅ Training Complete!\n")
print("-" * 20)
print("🎬 WATCHING THE SMART NPC IN ACTION:")

state = env.reset()
done = False
env.render()

while not done:
    time.sleep(0.5) 
    action = np.argmax(q_table[state[0], state[1], :]) 
    
    if action == 0:   move = "UP"
    elif action == 1: move = "DOWN"
    elif action == 2: move = "LEFT"
    elif action == 3: move = "RIGHT"
    
    print(f"NPC decides to move: {move}")
    state, reward, done = env.step(action)
    env.render()

if reward == 10:
    print("🏆 The Smart NPC successfully navigated the maze and found the target!")
