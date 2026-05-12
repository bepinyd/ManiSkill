import torch
import sapien
import numpy as np
from typing import Any, Union
from copy import deepcopy

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# --- STEP 1: Improved Agent Definition ---
@register_agent()
class LeapHandLeft(BaseAgent):
    uid = "my_leap_hand"
    urdf_path = "/home/bipin/ManiSkill/leap_hand/leap_hand_left.urdf"

    # Use numerical strings to ensure predictable action mapping
    joint_names = [str(i) for i in range(16)]
    
    @property
    def _controller_configs(self):
        # Define core parameters
        core_kwargs = dict(
            joint_names=self.joint_names,
            lower=None,   # <--- ADD THIS: Tells ManiSkill to use URDF limits
            upper=None,   # <--- ADD THIS: Tells ManiSkill to use URDF limits
            stiffness=200, 
            damping=20,
            force_limit=10,
            normalize_action=True,
        )
        
        return dict(
            # Absolute position controller
            pd_joint_pos=PDJointPosControllerConfig(**core_kwargs),
            
            # Delta position controller
            # For Delta, lower/upper are usually small (e.g. -0.1, 0.1) 
            # representing the max movement allowed per step.
            pd_joint_delta_pos=PDJointPosControllerConfig(
                joint_names=self.joint_names,
                stiffness=200, 
                damping=20,
                force_limit=10,
                lower=-0.1, # Max movement down per step
                upper=0.1,  # Max movement up per step
                use_delta=True
            )
        )
        
       

# --- STEP 2: Task Environment ---
@register_env("PickCubeLeap-v1", max_episode_steps=100)
class PickCubeLeapHandEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_leap_hand"]
    agent: LeapHandLeft

    def __init__(self, *args, robot_uids="my_leap_hand", **kwargs):
        # Configs for the environment objects
        self.cube_half_size = 0.02
        self.goal_thresh = 0.025
        self.cube_spawn_half_size = 0.1
        self.max_goal_height = 0.2
        
        # Camera positions (Adjusted for the Leap Hand's likely scale)
        self.human_cam_eye_pos = [0.6, -0.4, 0.4]
        self.human_cam_target_pos = [0, 0, 0]
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        
        # Build the Cube
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
        )
        
        # Build the Goal Site (Visual only)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.5],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return{}


    def _get_obs_extra(self, info: dict):
        return dict()




if __name__ == "__main__":
    # This part only runs if you do 'python3 empty_world.py'
    import gymnasium as gym
    
    # Create the environment we just registered above
    env = gym.make("PickCubeLeap-v1", render_mode="human",reward_mode="none"  )
    
    obs, _ = env.reset()
    for _ in range(1000):
        # Move the robot randomly
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()