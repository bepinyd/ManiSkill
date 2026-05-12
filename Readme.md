# Dexterous Grasping with LEAP Hand & ROS 2 Integration

This repository contains the implementation for the **Robotics Research Center (RRC), IIIT Hyderabad Research Internship Shortlisting Exercise**.

The project focuses on:
- Deep Reinforcement Learning for dexterous manipulation
- LEAP Hand grasping control
- ROS 2 integration for visualization and deployment

---

# 🎯 Project Objective

The objective is to train a **LEAP Hand** using **PPO (Proximal Policy Optimization)** to perform dexterous grasping of a cube in simulation.

The trained policy should:
- Establish stable multi-finger contact
- Lift the cube off the table
- Maintain grasp stability
- Stream joint trajectories to ROS 2
- Visualize hand motion in RViz2

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Simulator | ManiSkill / SAPIEN |
| RL Algorithm | PPO |
| RL Framework | CleanRL |
| Language | Python + C++ |
| Middleware | ROS 2 Humble |
| Visualization | RViz2 |

---

# 📂 Repository Structure

```bash
.
├── git/
├── leap_hand/
├── ppo/
├── runs/
├── ros2_mani/
│
├── leap_hand_exp.py
├── leap_hand_sim.py
├── ppo.py
│
└── ros2_mani/
    ├── src/
    │   └── mani_rviz/
    │       ├── CMakeLists.txt
    │       ├── package.xml
    │       ├── include/
    │       ├── launch/
    │       └── src/
```

---

# 🚀 Installation

## 1. Clone Repository

```bash
https://github.com/bepinyd/ManiSkill.git
cd ManiSkill
```

---

## 2. Create Python Environment

```bash
conda create -n mani_env python=3.10
conda activate mani_env
```

---

## 3. Install Dependencies

Install ManiSkill and reinforcement learning dependencies:

```bash
pip install mani_skill
pip install cleanrl
pip install torch torchvision
pip install gymnasium
pip install tensorboard
```

---

# 🤖 Task A — Learning-Based Dexterous Grasping

The PPO agent is trained in a custom ManiSkill environment:

```bash
PickCubeLeap-v1
```

---

# 🧠 Environment Description

## Observation Space

The observation vector contains:
- Joint positions
- Joint velocities
- Cube pose
- Relative fingertip positions
- End-effector information

---

## Action Space

Continuous **16-dimensional action space** representing:
- PD joint position targets
- LEAP Hand joint commands

---

# 🏆 Reward Formulation

The dense reward function consists of multiple components.

## Reach Reward
Encourages fingertips to approach the cube center.

## Contact Reward
Provides bonus rewards for stable multi-finger contact.

## Lift Reward
Rewards lifting the cube toward a target height.

## Stability Penalty
Penalizes horizontal displacement and unstable grasps.

## Action Penalty
Encourages smooth and energy-efficient control.

---

# ▶️ PPO Training

Run the following command to start PPO training:

```bash
python ppo.py \
    --env_id="PickCubeLeap-v1" \
    --num_envs=1 \
    --update_epochs=4 \
    --num_minibatches=4 \
    --num_steps=128 \
    --total_timesteps=50000 \
    --eval_freq=5 \
    --no-capture_video
```

---

# 📊 PPO Training Pipeline

```text
Reset Environment
       ↓
Collect Rollouts
       ↓
Compute Advantages
       ↓
PPO Optimization
       ↓
Policy Update
       ↓
Evaluation
```

---

# 🎮 Simulation Visualization

To visualize the environment without training:

```bash
python leap_hand_sim.py
```

This launches the ManiSkill/SAPIEN simulator and renders the LEAP Hand interacting with the cube.

---

# 🤖 Task B — ROS 2 Integration

The ROS 2 module streams joint commands from the learned policy for RViz2 visualization.

The ROS 2 package is implemented in **C++**.

---

# 📁 ROS 2 Package Structure

```bash
ros2_mani/src/mani_rviz/
├── CMakeLists.txt
├── package.xml
├── include/
├── launch/
└── src/
```

---

# ⚙️ Build ROS 2 Workspace

Navigate to the ROS 2 workspace:

```bash
cd ros2_mani
```

Build using colcon:

```bash
colcon build
```

Source the workspace:

```bash
source install/setup.bash
```

---

# 🖥️ Launch RViz2 Visualization

Launch the visualization pipeline:

```bash
ros2 launch mani_rviz display.launch.py
```

This starts:
- robot_state_publisher
- joint_state_publisher
- RViz2 visualization

---

# 🎯 Run Controller Node

Execute the ROS 2 controller node:

```bash
ros2 run mani_rviz leap_ctrl
```

The node publishes desired joint targets to:

```bash
/hand/joint_commands
```

---

# 📈 Training Outputs

Training logs and checkpoints are stored in:

```bash
runs/
```

These can be visualized using TensorBoard.

---

# 📊 TensorBoard

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

Open in browser:

```text
http://localhost:6006
```

---

# 🧩 Features

- PPO-based dexterous manipulation
- Dense reward shaping
- Custom ManiSkill environment
- ROS 2 C++ integration
- RViz2 visualization
- LEAP Hand URDF support

---

# 🔮 Future Improvements

Potential future work:
- Sim-to-real transfer
- Domain randomization
- Vision-based grasping
- Tactile sensing
- Diffusion policy integration
- Isaac Lab support

---

# 📚 Resources

## ManiSkill
https://maniskill.readthedocs.io/

## ROS 2
https://docs.ros.org/en/humble/index.html

## SAPIEN
https://sapien.ucsd.edu/

## CleanRL
https://docs.cleanrl.dev/

---

# 👨‍💻 Author

Developed for the:

**RRC IIIT Hyderabad Research Internship Shortlisting Exercise**

---

# 📜 License

This project is intended for educational and research purposes.
