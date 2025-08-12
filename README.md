# FloodIt GNN Agent

This project implements a **Graph Neural Network (GNN)-based reinforcement learning agent** to play the puzzle game *FloodIt*.
It leverages **Stable-Baselines3 (PPO)** with a custom GNN feature extractor for grid-based environments.
The project also includes a **Streamlit app** for interactive gameplay with the trained model.

---

## Project Structure

```
.
├── app.py              # Streamlit app for playing FloodIt with trained PPO-GNN
├── floodit_env.py      # Custom Gymnasium environment for FloodIt
├── train_gnn.py        # GNN feature extractor & PPO training logic
├── train_large.py      # Large-scale training script for PPO-GNN
├── play.py             # Script to play/evaluate model without Streamlit
├── test.py             # Testing utilities and evaluation scripts
└── requirements.txt    # Python dependencies (you should create this)
```

---

## Features

* **Custom FloodIt Environment**

  * Built using `gymnasium` for easy RL integration
  * Supports configurable board size, colors, and max steps

* **Graph Neural Network Feature Extractor**

  * Implements **GraphSAGE** layers for spatial reasoning over the grid
  * Converts board states into rich embeddings for the policy network

* **PPO Training**

  * Uses **Stable-Baselines3 PPO**
  * Parallelized training with `SubprocVecEnv` for speed

* **Interactive Gameplay**

  * Streamlit interface for visualizing board states and moves
  * Model evaluation in real-time

---

## Installation

```bash
git clone https://github.com/KaaustaaubShankar/Filler.git
cd Filler
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training

### Train GNN-based PPO Model

```bash
python train_large.py
```

or

```bash
python train_gnn.py
```

This will save the model to a `.zip` file (default: `ppo_floodit_gnn_vectorized_large.zip`).

---

## Play with the Model

### Streamlit Interface

```bash
streamlit run app.py
```

* Adjust board size, colors, and seeds in the UI.
* Visualize each step as the agent plays.

### CLI Play

```bash
python play.py
```

---

## Testing

```bash
python test.py
```

Evaluates the trained agent’s performance and logs results.

---

## How It Works

1. **Environment**
   `FloodItEnv` encodes the board as a grid of colors.
   The agent selects a color to flood-fill from the top-left corner.

2. **GNN Feature Extraction**

   * Each cell is treated as a node in a graph
   * Edges connect adjacent cells
   * GraphSAGE layers aggregate local neighborhood information

3. **PPO Agent**

   * Learns to minimize steps required to flood the entire board
   * Optimizes via reward signals from the environment

---

## Future Improvements

* Add self-play curriculum learning
* Experiment with different GNN architectures (e.g., GAT, GIN)
* Tune hyperparameters for larger boards

---

## License

MIT License – feel free to use and modify.
