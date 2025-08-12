import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FloodItEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=14, n_colors=6, max_steps=None, reward_dense=True, seed=None,evaluate_bool = False):
        super().__init__()
        self.size = size
        self.n_colors = n_colors
        self.reward_dense = reward_dense
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps or size * 2

        # Action space: pick one of the colors
        self.action_space = spaces.Discrete(self.n_colors)
        # Observation space: grid of color IDs
        self.observation_space = spaces.Box(
            low=0, high=self.n_colors - 1,
            shape=(self.size, self.size),
            dtype=np.int32
        )

        self.board = None
        self.flooded = None
        self.steps = 0
        self.current_color = None
        self.eval = evaluate_bool

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.board = self.rng.integers(0, self.n_colors, size=(self.size, self.size), dtype=np.int32)
        self.steps = 0
        self.current_color = self.board[0, 0]
        self._flood_fill(self.current_color)  # mark flooded region
        info = {}
        return self._get_obs(), info

    def step(self, action):
        action = int(action)
        terminated = False
        truncated = False

        flooded_before = self.flooded.copy()
        reward = 0.0

        self.current_color = action
        self._flood_fill(action)
        flooded_after = self.flooded
        gained = flooded_after.sum() - flooded_before.sum() #needs to gain if its going to move
        if gained  != 0:
            reward += 1 * gained #proportional to size isnt giving enough stimulus
        else:
            reward -= 4


        self.steps += 1

        if self.flooded.all():
            terminated = True
            reward += 5.0  # big win bonus
            if self.eval:
                print(f"win {reward - 5}")
        elif self.steps >= self.max_steps:
            truncated = True

        if terminated != True:
            reward -= 1 # penalize for taking a step
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return self.board.copy()

    def _flood_fill(self, new_color):
        H, W = self.size, self.size

        # ---- Pass 1: collect current region (old color) ----
        old_color = self.board[0, 0]
        flooded = np.zeros_like(self.board, dtype=bool)
        stack = [(0, 0)]
        flooded[0, 0] = True

        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not flooded[nx, ny]:
                    if self.board[nx, ny] == old_color:
                        flooded[nx, ny] = True
                        stack.append((nx, ny))

        # Repaint that region to the chosen color
        if new_color != old_color:
            self.board[flooded] = new_color

            # ---- Pass 2: immediately absorb adjacent cells that already have new_color ----
            # Start from (0,0) again and flood by new_color to avoid the 1-step lag.
            expanded = np.zeros_like(self.board, dtype=bool)
            stack = [(0, 0)]
            expanded[0, 0] = True

            while stack:
                x, y = stack.pop()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and not expanded[nx, ny]:
                        if self.board[nx, ny] == new_color:
                            expanded[nx, ny] = True
                            stack.append((nx, ny))

            flooded = expanded  # this is the true flooded region after the move

        # Keep board values consistent (no-op if already set)
        self.board[flooded] = new_color
        self.flooded = flooded

    def render(self):
        for row in self.board:
            print(" ".join(str(c) for c in row))
        print()
