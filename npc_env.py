import numpy as np

class SmartNPCEnv:
    def __init__(self):
        self.grid_size = 5
        self.reset()
        self.action_space = [0, 1, 2, 3]

    def reset(self):
        """Resets the environment to the starting state."""
        self.agent_pos = [0, 0] 
        self.goal_pos = [4, 4]   
        self.obstacles = [[1, 2], [2, 2], [3, 2]] 
        self.done = False
        return tuple(self.agent_pos)

    def step(self, action):
        """Applies an action and returns the new state, reward, and if done."""
        if self.done:
            return tuple(self.agent_pos), 0, True

        new_pos = list(self.agent_pos)
        if action == 0 and new_pos[0] > 0:            new_pos[0] -= 1 
        elif action == 1 and new_pos[0] < self.grid_size - 1: new_pos[0] += 1 
        elif action == 2 and new_pos[1] > 0:            new_pos[1] -= 1 
        elif action == 3 and new_pos[1] < self.grid_size - 1: new_pos[1] += 1 

        self.agent_pos = new_pos

        if self.agent_pos == self.goal_pos:
            self.done = True
            reward = 10  
        elif self.agent_pos in self.obstacles:
            self.done = True
            reward = -10
        else:
            reward = -1  

        return tuple(self.agent_pos), reward, self.done

    def render(self):
        """Prints the current state of the grid using emojis."""
        grid = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if [i, j] == self.agent_pos:
                    grid += "[🤖]"
                elif [i, j] == self.goal_pos:
                    grid += "[🏆]"
                elif [i, j] in self.obstacles:
                    grid += "[🌲]"
                else:
                    grid += "[⬜]"
            grid += "\n"
        print(grid)
        print("-" * 20)

if __name__ == "__main__":
    env = SmartNPCEnv()
    print("Initial State:")
    env.render()
