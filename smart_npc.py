import numpy as np
import random
import time
import google.generativeai as genai
from npc_env import SmartNPCEnv


USE_API = False 

API_KEY = "AIzaSyC94R3cf1y16Og5GJS48DtIYY5cq7ROmscE" 

if USE_API:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-8b') 

env = SmartNPCEnv()
q_table = np.zeros((env.grid_size, env.grid_size, len(env.action_space)))

print("🧠 Training the NPC Brain... (Running 500 simulations)")
learning_rate = 0.1
discount_factor = 0.9      
exploration_rate = 1.0     
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

for episode in range(500):
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
    exploration_rate = max(min_exploration_rate, exploration_rate * np.exp(-exploration_decay_rate * episode))

print("✅ Brain Trained!\n")
print("-" * 30)
print("🎬 THE NEURO-SYMBOLIC NPC IN ACTION:")

# --- THE DIALOGUE ENGINE ---
def generate_npc_dialogue(current_state, reward):
    """Handles both the Real AI Voice and the Backup Logic."""
    
    if not USE_API:
        if reward == 10: return "🏆 Goal reached! I am the ultimate algorithm!"
        if reward == -10: return "💥 Ouch! Who planted that tree there?!"
        if current_state == (0,0): return "🤖 Systems online. Pathfinding initialized..."
        return random.choice(["*beep*", "*whirrr*", "Calculating...", "Moving...", "Step taken."])

    if reward == 10:
        scenario = "You just successfully reached your goal! Generate a sarcastic one-liner."
    elif reward == -10:
        scenario = "You just crashed into a tree! Generate an annoyed robotic one-liner."
    elif current_state == (0,0):
        scenario = "You are at the start of a maze. Generate a determined robotic one-liner."
    else:
        scenario = "You are taking a step in a maze. Generate a 2-word robotic sound."

    prompt = f"Roleplay: Small sarcastic robot NPC. Under 10 words. Scenario: {scenario}"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"*bzzzt* Voice module glitch: {str(e)[:40]}..."

state = env.reset()
done = False
env.render()

print(f"💬 NPC: \"{generate_npc_dialogue(state, 0)}\"\n")

while not done:
    time.sleep(4.0) 
    action = np.argmax(q_table[state[0], state[1], :]) 
    
    state, reward, done = env.step(action)
    env.render()
    
    dialogue = generate_npc_dialogue(state, reward)
    print(f"💬 NPC: \"{dialogue}\"\n")

if reward == 10:
    print("🏆 Mission Accomplished!")
