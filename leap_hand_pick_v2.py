import torch
import sapien
import numpy as np
from typing import Any, Union

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

# =============================================================================
# 1. AGENT DEFINITION (The LEAP Hand)
# =============================================================================
@register_agent()
class LeapHandLeft(BaseAgent):
    uid = "my_leap_hand"
    urdf_path = "/home/bipin/ManiSkill/leap_hand/leap_hand_left.urdf"

    joint_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

    @property
    def _controller_configs(self):
        core_kwargs = dict(
            joint_names=self.joint_names,
            lower=None,
            upper=None,
            stiffness=200,
            damping=20,
            force_limit=10,
            normalize_action=True,
        )
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(**core_kwargs),
            pd_joint_delta_pos=PDJointPosControllerConfig(**core_kwargs, use_delta=True),
        )

    @property
    def tcp(self):
        """Tool Center Point: the palm link."""
        return self.robot.find_link_by_name("palm_lower_left")


# =============================================================================
# 2. ENVIRONMENT DEFINITION (The Pick Task)
# =============================================================================
@register_env("PickCubeLeap-v1", max_episode_steps=100)
class PickCubeLeapHandEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_leap_hand"]
    agent: LeapHandLeft

    def __init__(self, *args, robot_uids="my_leap_hand", **kwargs):
        self.cube_half_size = 0.02

        # --- POSITION TUNING ---
        # Hand base spawns at x=-0.4, z=0.15, facing +x (palm toward cube).
        # The URDF base_joint origin is at xyz="0.015 0.055 0.17", so the palm
        # extends forward ~0.17-0.25m from the base in the +x direction.
        # Fingertips reach roughly to x=-0.15, z~=0.0 (table surface level).
        # Cube is placed directly in that reachable zone.
        self.cube_pos = np.array([-0.4, 0.0, self.cube_half_size])  # on table surface, centered in front of hand
        self.goal_height = 0.15   # lift the cube 15cm above table
        self.goal_thresh = 0.05   # 5cm success radius (generous for early training)

        # Hand spawn position and orientation
        # z=0.15 keeps the palm ~15cm above table so fingers naturally hang toward the cube
        self.hand_spawn_pos  = [-0.42, 0.0, 0.15]
        self.hand_spawn_quat = [0.707, 0.0, 0.707, 0.0]  # 90deg around Y => palm faces +x

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        initial_pose = sapien.Pose(
            p=self.hand_spawn_pos,
            q=self.hand_spawn_quat,
        )
        super()._load_agent(options, initial_pose)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Cube at fixed, reachable position — no randomization
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=self.cube_pos.tolist()),
        )

        # Goal site: directly above the cube
        goal_pos = [self.cube_pos[0], self.cube_pos[1], self.goal_height]
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.4],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=goal_pos),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Same fixed cube position every episode — easier for policy to learn
            cube_xyz = torch.tensor(
                self.cube_pos, dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(b, -1).clone()
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

            # Goal directly above cube
            goal_xyz = torch.tensor(
                [self.cube_pos[0], self.cube_pos[1], self.goal_height],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).expand(b, -1).clone()
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            obj_pose=self.cube.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )

    def evaluate(self):
        obj_to_goal_dist = torch.linalg.norm(
            self.cube.pose.p - self.goal_site.pose.p, axis=1
        )
        success = obj_to_goal_dist < self.goal_thresh
        return {"success": success, "obj_to_goal_dist": obj_to_goal_dist}

    # -------------------------------------------------------------------------
    # GRASP DETECTION via contact forces
    # -------------------------------------------------------------------------
    def _is_grasping(self, obj) -> torch.Tensor:
        """
        Sums contact force magnitudes from all 4 fingertips against the cube.
        Returns float tensor (num_envs,): 1.0 if total force > threshold.
        """
        fingertip_link_names = [
            "fingertip",        # index
            "fingertip_2",      # middle
            "fingertip_3",      # ring
            "thumb_fingertip",  # thumb
        ]

        fingertip_links = []
        for name in fingertip_link_names:
            link = self.agent.robot.find_link_by_name(name)
            if link is not None:
                fingertip_links.append(link)

        total_force = torch.zeros(self.num_envs, device=self.device)
        for link in fingertip_links:
            try:
                forces = self.scene.get_pairwise_contact_forces(link, obj)
                total_force += torch.linalg.norm(forces, dim=-1)
            except Exception:
                pass  # no contact data yet at episode start

        force_threshold = 0.5  # Newtons — lower = more sensitive
        return (total_force > force_threshold).float()

    # -------------------------------------------------------------------------
    # REWARD LOGIC: Reach -> Grasp -> Lift
    # -------------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # 1. REACHING: reward palm for getting close to the cube
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_obj_dist)

        # 2. GRASPING: contact-based reward
        is_grasped = self._is_grasping(self.cube)
        grasp_reward = is_grasped * 2.0

        # 3. LIFTING: reward moving cube to goal height (only when grasped)
        obj_to_goal_dist = info["obj_to_goal_dist"]
        lifting_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_dist)) * is_grasped

        # 4. ACTION PENALTY: discourage large/jerky joint commands
        action_penalty = -0.01 * torch.linalg.norm(action, dim=-1)

        reward = reaching_reward + grasp_reward + lifting_reward + action_penalty

        # 5. SUCCESS BONUS
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 10.0


# =============================================================================
# 3. LAUNCHER (Test with Random Actions)
# =============================================================================
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("PickCubeLeap-v1", render_mode="human", reward_mode="dense")

    obs, _ = env.reset()
    print("Environment Loaded! Starting random simulation...")
    print(f"  Hand spawn pos : {env.unwrapped.hand_spawn_pos}")
    print(f"  Cube position  : {env.unwrapped.cube_pos}  (fixed, on table)")
    print(f"  Goal height    : {env.unwrapped.goal_height} m above table")
    print(f"  Success thresh : {env.unwrapped.goal_thresh} m")

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode Finished. Success: {info['success'].any().item()}")
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("PickCubeLeap-v1", render_mode="human", reward_mode="dense")

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