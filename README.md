Project Title & Description

NKPL-Underwater-Defense is a hybrid algorithm combining Neural networks, Kalman filtering, and Path Learning (RL-based planning) for autonomous underwater defense navigation.
It solves the challenge of safe and efficient underwater vehicle navigation in environments with obstacles, noise, and uncertainty, where traditional static planners like A* fail to adapt.

⚙️ System Overview

The system integrates three core components that work together at runtime:

Reinforcement Learning (RL): Learns dynamic action policies for path optimization.

Kalman Filter: Filters noisy sonar/sensor measurements and provides accurate state estimation.

Convolutional Neural Network (CNN): Processes raw sonar images or environment grids for obstacle detection and environmental awareness.

The workflow:

CNN extracts environment features from sonar-like input.

Kalman filter stabilizes the predicted state under noise.

RL agent receives the cleaned state and selects optimal actions (move left, right, forward, etc.).

Feedback loop continues until the goal is reached.

Installation

Python version: ≥ 3.8

Clone repository:git clone https://github.com/yourusername/NKPL-Underwater-Defense.git
cd NKPL-Underwater-Defense

nstall dependencies:
pip install -r requirements.txt

Dependencies (inside requirements.txt):
torch
torchvision
tensorflow
numpy
matplotlib
scipy

▶️ How to Run
1. Training the RL agent
   python src/train.py --episodes 500 --lr 0.001 --gamma 0.95

Running simulations
python src/simulate.py --scenario synthetic_data_1.json

Testing with pretrained models
python src/test.py --model models/knpl_pretrained.pth
💻 Example Usage

Run a full pipeline with training + simulation in one go:

bash scripts/run_experiment.sh


Or visualize navigation results:

python src/visualize.py --log runs/exp1/

git init
git add .
git commit -m "Initial commit: NKPL repo structure"
git branch -M main
git remote add origin https://github.com/yourusername/NKPL-Underwater-Defense.git
git push -u origin main
