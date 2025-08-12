import io
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from floodit_env import FloodItEnv
from train_gnn import GridGNNExtractor

# -----------------------------
# Config (locked values)
# -----------------------------
BOARD_SIZE = 12
NUM_COLORS = 4
MAX_STEPS = 200
MODEL_PATH = "ppo_floodit_gnn_vectorized_large.zip"  # change if needed

# -----------------------------
# Helpers
# -----------------------------
def make_env(size=BOARD_SIZE, n_colors=NUM_COLORS, max_steps=MAX_STEPS, seed=None, evaluate=True):
    return FloodItEnv(size=size, n_colors=n_colors, max_steps=max_steps,
                      seed=seed, evaluate_bool=evaluate)

def color_palette(n):
    base = np.array([
        [231, 76, 60],
        [46, 204, 113],
        [52, 152, 219],
        [241, 196, 15],
        [155, 89, 182],
        [26, 188, 156],
        [230, 126, 34],
        [149, 165, 166],
    ], dtype=np.float32) / 255.0
    reps = int(np.ceil(n / len(base)))
    return np.vstack([base] * reps)[:n]

def draw_board(board, palette, title=None):
    h, w = board.shape
    img = palette[board]
    fig, ax = plt.subplots(figsize=(w/2.5, h/2.5))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def reset_episode(seed=None):
    st.session_state.env = make_env(seed=seed, evaluate=True)
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.total_reward = 0.0
    st.session_state.step = 0
    st.session_state.last_reward = 0.0

def predict_step(deterministic=True):
    env = st.session_state.env
    obs = st.session_state.obs
    model = st.session_state.model

    action, _ = model.predict(obs, deterministic=deterministic)
    obs, reward, terminated, truncated, info = env.step(action)
    st.session_state.obs = obs
    st.session_state.last_reward = float(reward)
    st.session_state.total_reward += float(reward)
    st.session_state.step += 1
    st.session_state.done = bool(terminated or truncated)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Flood-It RL Viewer", layout="centered")
st.title("🧠 Flood-It RL Viewer")

with st.sidebar:
    st.header("Environment Settings")
    st.markdown(f"**Board size:** {BOARD_SIZE} × {BOARD_SIZE}")
    st.markdown(f"**Colors:** {NUM_COLORS}")
    st.markdown(f"**Max steps:** {MAX_STEPS}")

    deterministic = st.checkbox("Deterministic actions", value=True)
    autoplay = st.checkbox("Auto-play", value=False)
    speed = st.slider("Auto-play speed (sec/step)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    reset_btn = st.button("Reset Episode")
    step_btn = st.button("Step")

# Init session state
if "env" not in st.session_state:
    reset_episode(seed=None)
if "palette" not in st.session_state:
    st.session_state.palette = color_palette(NUM_COLORS)
if "model" not in st.session_state:
    try:
        st.session_state.model = PPO.load(MODEL_PATH, env=st.session_state.env, device="auto", print_system_info=False)
        st.success("Model loaded ✔")
    except Exception as e:
        st.session_state.model = None
        st.error(f"Failed to load model: {e}")

# Reset
if reset_btn:
    seed_val = None
    if seed_in.strip():
        try:
            seed_val = int(seed_in)
        except ValueError:
            st.warning("Seed must be an integer; ignoring.")
            seed_val = None
    reset_episode(seed=seed_val)

# -----------------------------
# Placeholders + render function
# -----------------------------
ph_metrics = st.empty()
ph_board = st.empty()

def render_frame():
    with ph_metrics.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Step", st.session_state.step)
        c2.metric("Last reward", f"{st.session_state.last_reward:.3f}")
        c3.metric("Total reward", f"{st.session_state.total_reward:.3f}")
        c4.metric("Done", "✅" if st.session_state.done else "—")

    with ph_board.container():
        board = st.session_state.env.board.copy()
        title = "Final board" if st.session_state.done else "Current board"
        draw_board(board, st.session_state.palette, title=title)

# Manual single step
if step_btn and not st.session_state.done:
    predict_step(deterministic=deterministic)
    render_frame()

# Auto-play
if autoplay and not st.session_state.done:
    max_more_steps = MAX_STEPS - st.session_state.step
    for _ in range(max(1, max_more_steps)):
        if st.session_state.done:
            break
        predict_step(deterministic=deterministic)
        render_frame()
        if speed > 0:
            time.sleep(speed)
    st.stop()

# Render current frame if not autoplaying
render_frame()

# Optional legend
with st.expander("Legend / Colors"):
    for i in range(NUM_COLORS):
        r, g, b = (st.session_state.palette[i] * 255).astype(int)
        st.markdown(
            f"<div style='display:inline-block;width:16px;height:16px;background:rgb({r},{g},{b});"
            f"border-radius:3px;margin-right:8px;vertical-align:middle;'></div> Color {i}",
            unsafe_allow_html=True,
        )

st.caption("This demo runs a fixed 12×12 Flood-It environment with a pre-trained PPO+GNN model.")
