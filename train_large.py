import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym
from floodit_env import FloodItEnv

# -----------------------------
# GraphSAGE layer (mean aggregation)
# -----------------------------
class GraphSageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.W_nei  = nn.Linear(in_dim, out_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, h: th.Tensor, A: th.Tensor, deg: th.Tensor) -> th.Tensor:
        # h: (B,N,F), A: (N,N), deg: (N,1)
        agg = th.einsum('ij,bjf->bif', A, h) / deg.unsqueeze(0)  # mean neighbors
        out = self.W_self(h) + self.W_nei(agg)
        return self.act(out)

# -----------------------------
# GNN feature extractor
# -----------------------------
class GridGNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        embed_dim: int = 32,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        features_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 2, "Expect (H, W) grid"
        H, W = observation_space.shape
        N = H * W

        n_colors = int(np.max(observation_space.high)) + 1

        # Create grid adjacency matrix
        A = th.zeros((N, N), dtype=th.float32)
        def idx(x, y): return x * W + y
        for x in range(H):
            for y in range(W):
                i = idx(x, y)
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        A[i, idx(nx, ny)] = 1.0
        deg = A.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("A", A)
        self.register_buffer("deg", deg)

        # Positional encoding
        xs = th.arange(H, dtype=th.float32).unsqueeze(1).repeat(1, W)
        ys = th.arange(W, dtype=th.float32).unsqueeze(0).repeat(H, 1)
        pos = th.stack([xs / (H-1 if H > 1 else 1), ys / (W-1 if W > 1 else 1)], dim=-1)
        self.register_buffer("pos", pos.view(N, 2))

        self.color_emb = nn.Embedding(n_colors, embed_dim)

        in_dim = embed_dim + 1 + 2  # color_emb + flooded_flag + (x,y)
        self.gnn = nn.ModuleList([
            GraphSageLayer(in_dim if i == 0 else gnn_hidden, gnn_hidden)
            for i in range(gnn_layers)
        ])

        # Pooling layers
        pooled_dim = gnn_hidden * 2
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim
        self.H, self.W, self.N = H, W, N
        self.n_colors = n_colors

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def _bfs_flooded(self, grid_long: th.Tensor) -> th.Tensor:
        """Compute flooded region using BFS"""
        H, W, N = self.H, self.W, self.N
        start_color = int(grid_long[0, 0].item())
        flooded = th.zeros((H, W), dtype=th.bool, device=grid_long.device)
        flooded[0, 0] = True
        stack = [(0, 0)]
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not flooded[nx, ny]:
                    if int(grid_long[nx, ny].item()) == start_color:
                        flooded[nx, ny] = True
                        stack.append((nx, ny))
        return flooded.view(N)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B, H, W = obs.shape
        device = obs.device

        idx = obs.round().clamp_(0, self.n_colors - 1).to(th.long)
        emb = self.color_emb(idx.view(B, -1))

        flooded_list = []
        for b in range(B):
            flooded_list.append(self._bfs_flooded(idx[b]))
        flooded = th.stack(flooded_list, dim=0).float()

        pos = self.pos.unsqueeze(0).expand(B, -1, -1).to(device)

        # Node features
        h = th.cat([emb, flooded.unsqueeze(-1), pos], dim=-1)

        # GNN processing
        A, deg = self.A.to(device), self.deg.to(device)
        for layer in self.gnn:
            h = layer(h, A, deg)

        # Pooling
        mean_all = h.mean(dim=1)
        denom_f = flooded.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mean_flooded = (h * flooded.unsqueeze(-1)).sum(dim=1) / denom_f
        g = th.cat([mean_all, mean_flooded], dim=-1)

        return self.proj(g)

# -----------------------------
# Environment factory with proper seeding
# -----------------------------
def make_env(rank, eval_mode=False):
    """Create environment with unique random seed per instance"""
    def _init():
        # Generate unique seed for each environment
        seed = np.random.randint(0, 2**32 - 1)
        env = FloodItEnv(
            size=12,
            n_colors=4,
            max_steps=200,
            seed=seed,
            evaluate_bool=eval_mode
        )
        return env
    return _init

def main():
    # Set up parallel environments
    num_cpu = 8  # Number of CPU cores to use
    print(f"Using {num_cpu} parallel environments")
    
    # Create training environments with unique seeds
    train_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Create evaluation environments
    eval_env = SubprocVecEnv([make_env(i + 1000, eval_mode=True) 
                            for i in range(min(4, num_cpu))])
    
    # Policy configuration
    policy_kwargs = dict(
        features_extractor_class=GridGNNExtractor,
        features_extractor_kwargs=dict(
            embed_dim=32,
            gnn_hidden=128,
            gnn_layers=3,
            features_dim=512,
        ),
        net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],
        activation_fn=nn.ReLU,
    )

    # PPO configuration for vectorized envs
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        batch_size=2048,           # Increased from 128*4
        n_steps=1024,              # Steps per env before update
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range_vf=0.2,
        max_grad_norm=0.5,
        n_epochs=10,               # Optimization epochs
        device="auto",             # Use GPU if available
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=max(10_000 // num_cpu, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    # Training
    print("Starting training with multi-seed vectorized environments...")
    total_timesteps = 500_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    model.save("ppo_floodit_gnn_vectorized_large")
    print(f"Training completed! Saved model with {total_timesteps} timesteps")

    # Final evaluation
    print("\nRunning final evaluation on unseen boards...")
    test_env = SubprocVecEnv([make_env(9999 + i, eval_mode=True) for i in range(10)])
    total_rewards = []
    total_steps = []
    
    obs = test_env.reset()
    dones = [False] * 10
    while not all(dones):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, new_dones, infos = test_env.step(actions)
        
        for i, done in enumerate(new_dones):
            if done and not dones[i]:
                # Episode completed
                total_rewards.append(infos[i].get("episode")["r"])
                total_steps.append(infos[i].get("episode")["l"])
        dones = new_dones
    
    # Calculate statistics
    win_rate = sum(1 for r in total_rewards if r > 0) / len(total_rewards)
    avg_moves = np.mean(total_steps) if total_steps else 0
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    
    print("\nFinal Evaluation Results:")
    print(f"  Boards tested: {len(total_rewards)}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Average moves: {avg_moves:.1f}")
    print(f"  Average reward: {avg_reward:.1f}")

if __name__ == "__main__":
    main()