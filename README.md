# BeerGame DQN Implementation

This repository provides a Deep Q-Network (DQN) and Double DQN implementation for the Beer Game simulation. The Beer Game is a classic supply chain management simulation designed to teach the dynamics of inventory management and order fulfillment. This project demonstrates how reinforcement learning can be applied to optimize supply chain operations.

## 📂 Project Structure

├── figures/ # Saved training figures and visualizations

├── models/ # Saved trained models

├── README.md # Project documentation

├── compare.py # Script to compare DQN and Double DQN performance

├── course_dqn_example.py # DQN training script

├── course_double_dqn.py # Double DQN training script


## 🚀 Features

- ✅ Basic DQN implementation for the Beer Game
- ✅ Double DQN to reduce overestimation bias
- ✅ Training and evaluation for both models
- ✅ Performance comparison with visualization
- ✅ Model checkpoint saving and loading

## 🛠️ Dependencies

- Python >= 3.8
- NumPy
- Matplotlib
- PyTorch
- gym (if using a Gym-style BeerGame environment)

## 🧠 Usage

1️⃣ Train with DQN
```
python course_dqn_example.py
```
2️⃣ Train with Double DQN
```
python course_double_dqn.py
```
3️⃣ Compare DQN and Double DQN performance
```
python compare.py
```
4️⃣ Output
Training logs and plots are saved in the figures/ folder.
Trained models are saved in the models/ folder.

