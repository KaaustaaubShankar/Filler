import time
import numpy as np
from floodit_env import FloodItEnv

# -------- Manual play loop --------
def play_manual():
    env = FloodItEnv(size=14, n_colors=6, max_steps=200, seed=42)
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        print(f"Step {info.get('step', 0)}, Reward so far: {total_reward}")
        try:
            action = int(input(f"Pick a color [0-{env.n_colors - 1}]: "))
        except ValueError:
            print("Please enter a valid integer")
            continue
        if action < 0 or action >= env.n_colors:
            print("Invalid action")
            continue
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.render()
    print("Game over! Total reward:", total_reward)

# -------- Random agent demo --------
def play_random(delay=0.3):
    env = FloodItEnv(size=14, n_colors=6, max_steps=200, seed=42)
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        print(f"Randomly picked action {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(delay)

    env.render()
    print("Game over! Total reward:", total_reward)

# -------- Trained model demo --------
def play_trained(model_path, delay=0.3):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = FloodItEnv(size=14, n_colors=6, max_steps=200, seed=123, evaluate_bool=True)

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        print(f"Model picked action {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(delay)

    env.render()
    print("Game over! Total reward:", total_reward)

if __name__ == "__main__":
    play_manual()
