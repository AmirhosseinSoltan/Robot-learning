# Learning-Based Robotic Manipulation with Kinova Gen3 7-DOF Arm

This repository contains code, configurations, and documentation for training a Kinova Gen3 7-DOF robotic arm (with Robotiq 2F-85 gripper) to visually locate, track, and manipulate a task board using a two-phase learning pipeline: behavior cloning from expert demonstrations, followed by reinforcement learning with PPO.

---

## 🚀 Features

- **Multi-modal perception**  
  - RGB camera input (84×84×3)  
  - Proprioceptive feedback (end-effector position/orientation, time)

- **Two-phase learning**  
  1. **Behavior Cloning** (BC) using expert ROS-bag demonstrations  
  2. **Proximal Policy Optimization** (PPO) fine-tuning in Gazebo simulation

- **Custom Gymnasium Environment**  
  - ROS Noetic ↔ Gazebo integration  
  - Observation & action spaces tailored for visual-servo control  
  - Reward shaping on visibility, smoothness, and energy efficiency

- **Modular, reusable code**  
  - CNN + FC policy network  
  - Clear separation: data collection, BC training, RL training, evaluation

---

## 📂 Repository Structure

```
.
├── configs/                 # YAML configs for BC and PPO training
├── data/
│   ├── demonstrations/      # ROS-bag files of expert trajectories
│   └── task_board_template/ # Template image(s) for board detection
├── envs/                    # Custom Gymnasium environment (KinovaEnv)
├── notebooks/               # Jupyter notebooks for analysis & visualization
├── scripts/
│   ├── collect_demos.py     # Record expert demos into ROS bags
│   ├── train_bc.py          # Train behavior cloning policy
│   └── train_ppo.py         # Run PPO training loop
├── src/
│   ├── models/              # Policy & value network definitions
│   ├── utils/               # CV utilities, reward functions, ROS wrappers
│   └── evaluation.py        # Test and visualize trained policies
├── results/                 # Training logs, TensorBoard snapshots, checkpoints
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## 🛠 Prerequisites

- **Hardware**  
  - Ubuntu 20.04 LTS  
  - ROS Noetic  
  - Gazebo 11  
  - (Optional) Real Kinova Gen3 7-DOF arm & Robotiq 2F-85 gripper

- **Software**  
  - Python 3.8+  
  - [ros_kortex](https://github.com/Kinovarobotics/ros_kortex)  
  - [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)  
  - [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)  
  - OpenCV, NumPy, PyTorch or TensorFlow (as per config)  

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Clone & build ROS packages:
```bash
# in your ROS workspace
git clone https://github.com/Kinovarobotics/ros_kortex.git src/ros_kortex
git clone https://github.com/hrii-iit/robothon-2023-board-description.git src/task_board_description
catkin_make
source devel/setup.bash
```

---

## 🔧 Installation & Setup

1. **Download Dataset & Assets**  
   - Expert demonstrations (ROS bags)  
   - Task-board template images  
   - [Google Drive folder →](https://drive.google.com/drive/folders/1TplSTFsR3e5zsR9gBBtWotSk0qjrH-oY?usp=sharing)

2. **Configure Parameters**  
   - Edit `configs/bc.yaml` for behavior cloning (learning rate, batch size, epochs).  
   - Edit `configs/ppo.yaml` for reinforcement learning (clip ratio, reward weights).

3. **Build & Launch Simulation**  
   ```bash
   # Launch Gazebo world with textured walls and task board
   roslaunch ros_kortex kortex_gazebo.launch world:=worlds/task_board_world.world
   ```

---

## ▶️ Usage

### 1. Collect Expert Demonstrations
```bash
python scripts/collect_demos.py   --output_dir data/demonstrations   --num_episodes 50
```

### 2. Train Behavior Cloning Policy
```bash
python scripts/train_bc.py   --config configs/bc.yaml   --demo_dir data/demonstrations   --output_dir results/bc_model
```

### 3. Fine-tune with PPO
```bash
python scripts/train_ppo.py   --config configs/ppo.yaml   --policy_checkpoint results/bc_model/best.pt   --output_dir results/ppo_model
```

### 4. Evaluate & Visualize
```bash
python src/evaluation.py   --model_checkpoint results/ppo_model/best.pt   --num_episodes 10   --save_videos True
```

---

## 📈 Results

- **Behavior Cloning**  
  - Converged on Huber loss ≈ 1e-3 after 200 epochs  
  - Demonstrated smooth tracking in validation scenarios  

- **PPO Fine-tuning**  
  - Stable improvement in visibility reward (> 0.8 match score)  
  - Learned smooth, energy-efficient motions  

Plots and example videos are available in `results/`.

---

## 🔬 Methodology & References

1. **Behavior Cloning Architecture**  
   - Adapted from Hermann _et al._, “Adaptive curriculum generation from demonstrations for sim-to-real visuomotor control”  
2. **Reinforcement Learning**  
   - PPO implementation via Stable-Baselines3  
3. **Gymnasium Environment**  
   - Custom `KinovaEnv` interfacing ROS & Gazebo  
4. **Task Board Detection**  
   - ORB keypoint matching & adaptive thresholding for visibility reward

For full bibliographic details, see [References](#references).

---

## 👥 Contributing

Contributions are welcome! Please open an issue or pull request to suggest improvements, report bugs, or add new features.

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📚 References

- Hermann, L., Argus, M., Eitel, A., Amiranashvili, A., Burgard, W., & Brox, T. (2019). Adaptive curriculum generation from demonstrations for sim-to-real visuomotor control. _CoRR_, abs/1910.07972.  
- Raffin, A., Hill, A., Ernestus, M., Gleave, A., Kanervisto, A., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. _GitHub_.  
- Chaurasia, Arunima. (2025). Kinova arm PPO training in Gazebo simulation. _YouTube_.