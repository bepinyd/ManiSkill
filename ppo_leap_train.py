import torch
import sapien
import numpy as np
from typing import Any, Union

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_agent()
class LeapHandLeft(BaseAgent):
    uid = "my_leap_hand"
    urdf_path = "/home/bipin/ManiSkill/leap_hand/leap_hand_left.urdf"
    joint_names = [str(i) for i in range(16)]

    @property
    def _controller_configs(self):
        core_kwargs = dict(
            joint_names=self.joint_names,
            lower=None, upper=None,
            stiffness=200, damping=20, force_limit=10,
            normalize_action=True,
        )
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(**core_kwargs),
            pd_joint_delta_pos=PDJointPosControllerConfig(**core_kwargs, use_delta=True),
        )

    @property
    def tcp(self):
        return self.robot.find_link_by_name("palm_lower_left")

@register_env("PickCubeLeap-v2", max_episode_steps=100)
class PickCubeLeapHandEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_leap_hand"]
    agent: LeapHandLeft

    def __init__(self, *args, robot_uids="my_leap_hand", **kwargs):
        self.cube_half_size = 0.02
        self.cube_pos = np.array([-0.4, 0.0, self.cube_half_size])
        self.goal_height = 0.15
        self.goal_thresh = 0.05
        
        # Camera Setup (Crucial for evaluation videos)
        self.human_cam_eye_pos = [0.2, -0.4, 0.4]
        self.human_cam_target_pos = [-0.4, 0, 0]
        
        self.hand_spawn_pos  = [-0.43, 0.0, 0.15]
        self.hand_spawn_quat = [0.707, 0.0, 0.707, 0.0]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        initial_pose = sapien.Pose(p=self.hand_spawn_pos, q=self.hand_spawn_quat)
        super()._load_agent(options, initial_pose)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.cube = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube",
                                     initial_pose=sapien.Pose(p=self.cube_pos.tolist()))
        self.goal_site = actors.build_sphere(self.scene, radius=self.goal_thresh, color=[0, 1, 0, 0.4], name="goal_site",
                                            body_type="kinematic", add_collision=False,
                                            initial_pose=sapien.Pose(p=[self.cube_pos[0], self.cube_pos[1], self.goal_height]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            cube_xyz = torch.tensor(self.cube_pos, device=self.device).repeat(b, 1)
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            obj_pose=self.cube.pose.raw_pose,
        )

    def evaluate(self):
        dist = torch.linalg.norm(self.cube.pose.p - self.goal_site.pose.p, axis=1)
        return {"success": dist < self.goal_thresh, "obj_to_goal_dist": dist}

    def _is_grasping(self):
        """
        Check if the fingertips are applying force to the cube.
        We iterate through the links because an Articulation is not a single body.
        """
        # 1. Identify the links we want to check
        # These names must match your URDF exactly
        fingertip_names = ["index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head"]
        
        total_force = torch.zeros(self.num_envs, device=self.device)
        
        for name in fingertip_names:
            link = self.agent.robot.find_link_by_name(name)
            if link is not None:
                # get_pairwise_contact_forces works on Links, not Articulations
                forces = self.scene.get_pairwise_contact_forces(link, self.cube)
                # forces shape: (num_envs, 3)
                total_force += torch.linalg.norm(forces, dim=-1)
        
        # 2. Return 1.0 if the total force is above a small threshold (e.g., 0.5 Newtons)
        return (total_force > 0.5).float()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp.pose.p, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)
        
        is_grasped = self._is_grasping()
        grasp_reward = is_grasped * 2.0
        
        lift_reward = (1.0 - torch.tanh(5.0 * info["obj_to_goal_dist"])) * is_grasped
        
        reward = reaching_reward + grasp_reward + lift_reward
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 10.0

if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("PickCubeLeap-v2", render_mode="human", reward_mode="dense")

    obs, _ = env.reset()
    print("Environment Loaded! Starting random simulation...")

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode Finished. Success: {info['success'].any().item()}")
            obs, _ = env.reset()

    env.close()