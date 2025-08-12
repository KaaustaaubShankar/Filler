import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from floodit_env import FloodItEnv  # unchanged env


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
    """
    Obs: integer grid (H,W) as float32 (SB3), values in [0, n_colors-1].
    Node features = color embedding || flooded_flag || (x_norm, y_norm).
    Pooling = concat(mean_all_nodes, mean_flooded_nodes) -> MLP -> features_dim.
    """
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

        # Dense adjacency (small N → fine)
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

        # (x,y) / (H-1,W-1)
        xs = th.arange(H, dtype=th.float32).unsqueeze(1).repeat(1, W)
        ys = th.arange(W, dtype=th.float32).unsqueeze(0).repeat(H, 1)
        pos = th.stack([xs / (H-1 if H > 1 else 1), ys / (W-1 if W > 1 else 1)], dim=-1)  # HxWx2
        self.register_buffer("pos", pos.view(N, 2))

        self.color_emb = nn.Embedding(n_colors, embed_dim)

        in_dim = embed_dim + 1 + 2  # color_emb + flooded_flag + (x,y)
        self.gnn = nn.ModuleList([GraphSageLayer(in_dim if i == 0 else gnn_hidden, gnn_hidden)
                                  for i in range(gnn_layers)])

        # pooled = mean_all || mean_flooded
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
        """Return flooded mask (N,) bool from (H,W) long grid, matching env logic."""
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
        # obs: (B,H,W) float32 with integer values
        B, H, W = obs.shape
        device = obs.device

        idx = obs.round().clamp_(0, self.n_colors - 1).to(th.long)  # (B,H,W)
        emb = self.color_emb(idx.view(B, -1))                        # (B,N,E)

        flooded_list = []
        for b in range(B):
            flooded_list.append(self._bfs_flooded(idx[b]))
        flooded = th.stack(flooded_list, dim=0).float()              # (B,N)

        pos = self.pos.unsqueeze(0).expand(B, -1, -1).to(device)     # (B,N,2)

        # node features
        h = th.cat([emb, flooded.unsqueeze(-1), pos], dim=-1)        # (B,N,F)

        # GNN message passing
        A, deg = self.A.to(device), self.deg.to(device)
        for layer in self.gnn:
            h = layer(h, A, deg)                                     # (B,N,Hdim)

        # Pool
        eps = 1e-8
        mean_all = h.mean(dim=1)                                     # (B,Hdim)
        denom_f = flooded.sum(dim=1, keepdim=True).clamp_min(eps)
        mean_flooded = (h * flooded.unsqueeze(-1)).sum(dim=1) / denom_f
        g = th.cat([mean_all, mean_flooded], dim=-1)                 # (B,2*Hdim)

        return self.proj(g)                                          # (B,features_dim)


# -----------------------------
# Env factory
# -----------------------------
def make_env(seed=None, evaluate=False):
    return FloodItEnv(size=6, n_colors=4, max_steps=30, seed=seed, evaluate_bool=evaluate)


def main():
    train_env = make_env()
    eval_env = make_env(evaluate=True)

    policy_kwargs = dict(
        features_extractor_class=GridGNNExtractor,
        features_extractor_kwargs=dict(
            embed_dim=32,
            gnn_hidden=128,
            gnn_layers=3,
            features_dim=512,
        ),
        net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],  # your bigger heads
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        batch_size=128,
        n_steps=4096,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range_vf=0.2,
        max_grad_norm=0.5,
        # target_kl=0.02,  # optional if KL spikes
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    print("Starting training (GNN, no frontier)...")
    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("ppo_floodit_gnn_nofrontier")
    print("Training finished!")

    # quick test rollout
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"total_reward: {total_reward}")

if __name__ == "__main__":
    main()
