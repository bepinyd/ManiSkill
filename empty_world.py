import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env

# --- STEP 1: Define and Register the Agent (The LEAP Hand) ---
@register_agent()
class LeapHandLeft(BaseAgent):
    uid = "my_leap_hand" # This is the ID used to load the robot
    urdf_path = "/home/bipin/ManiSkill/leap_hand/leap_hand_left.urdf"

    @property
    def _controller_configs(self):
        # You MUST define how the robot is controlled
        # This is a basic position controller for all joints
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                joint_names=[j.name for j in self.robot.get_active_joints()],
                lower=None,
                upper=None,
                stiffness=100,
                damping=10,
            )
        )

# --- STEP 2: Define and Register the Environment ---
@register_env("Empty-Leap", max_episode_steps=200000)
class EmptyEnv(BaseEnv):
    # Tell the environment which robots it can support
    SUPPORTED_ROBOTS = ["my_leap_hand"] 

    def __init__(self, *args, robot_uids="my_leap_hand", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Load the robot at the world origin
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        # Standard collision bit for the ground
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return dict()

if __name__ == "__main__":
    # This part only runs if you do 'python3 empty_world.py'
    import gymnasium as gym
    
    # Create the environment we just registered above
    env = gym.make("Empty-Leap", render_mode="human",reward_mode="none"  )
    
    obs, _ = env.reset()
    for _ in range(1000):
        # Move the robot randomly
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()