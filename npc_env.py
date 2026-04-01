import numpy as np

class SmartNPCEnv:
    def __init__(self):
        self.grid_size = 5
        self.reset()
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = [0, 1, 2, 3]

    def reset(self):
        """Resets the environment to the starting state."""
        self.agent_pos = [0, 0]  # Top-left
        self.goal_pos = [4, 4]   # Bottom-right
        self.obstacles = [[1, 2], [2, 2], [3, 2]] # A wall in the middle
        self.done = False
        return tuple(self.agent_pos)

    def step(self, action):
        """Applies an action and returns the new state, reward, and if done."""
        if self.done:
            return tuple(self.agent_pos), 0, True

        # Calculate new position
        new_pos = list(self.agent_pos)
        if action == 0 and new_pos[0] > 0:            new_pos[0] -= 1 # Up
        elif action == 1 and new_pos[0] < self.grid_size - 1: new_pos[0] += 1 # Down
        elif action == 2 and new_pos[1] > 0:            new_pos[1] -= 1 # Left
        elif action == 3 and new_pos[1] < self.grid_size - 1: new_pos[1] += 1 # Right

        self.agent_pos = new_pos

        # Check conditions and assign rewards
        if self.agent_pos == self.goal_pos:
            self.done = True
            reward = 10  # Found the goal!
        elif self.agent_pos in self.obstacles:
            self.done = True
            reward = -10 # Hit a wall!
        else:
            reward = -1  # Small penalty to encourage the shortest path

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

# Quick test to see the grid
if __name__ == "__main__":
    env = SmartNPCEnv()
    print("Initial State:")
    env.render()