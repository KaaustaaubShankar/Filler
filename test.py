import time
from stable_baselines3 import PPO
from floodit_env import FloodItEnv
from train_gnn import GridGNNExtractor

# Load the environment (must match training params)
env = FloodItEnv(size=6, n_colors=4, max_steps=30, evaluate_bool=True)

# Load the trained model
model = PPO.load("ppo_floodit_gnn_nofrontier.zip", env=env, device="cpu")

# Run one episode
obs, info = env.reset()
done = False
total_reward = 0.0
step = 0

while not done:
    print(f"Step {step}, total_reward: {total_reward:.3f}")
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    done = terminated or truncated
    time.sleep(0.3)

print("\nFinal board:")
env.render()
print(f"Episode finished in {step} steps with total reward {total_reward:.3f}")
