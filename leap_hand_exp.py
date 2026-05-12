import torch
import sapien
import numpy as np
from typing import Any

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# =============================================================================
# 1. AGENT DEFINITION
# =============================================================================
@register_agent()
class LeapHandLeft(BaseAgent):
    uid = "my_leap_hand"
    urdf_path = "/home/bipin/ManiSkill/leap_hand/leap_hand_left.urdf"

    joint_names = [
        '0','1','2','3','4','5','6','7',
        '8','9','10','11','12','13','14','15'
    ]

    @property
    def _controller_configs(self):
        core_kwargs = dict(
            joint_names=self.joint_names,
            lower=None,
            upper=None,
            stiffness=50,
            damping=10,
            force_limit=5,
            normalize_action=True,
        )
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(**core_kwargs),
            pd_joint_delta_pos=PDJointPosControllerConfig(
                **core_kwargs, use_delta=True
            ),
        )

    @property
    def tcp(self):
        return self.robot.find_link_by_name("palm_lower_left")


# =============================================================================
# 2. ENVIRONMENT
# =============================================================================
@register_env("PickCubeLeap-v1", max_episode_steps=200)
class PickCubeLeapHandEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_leap_hand"]
    agent: LeapHandLeft

    # fingertip link names for contact detection & fingertip reward
    FINGERTIP_NAMES = [
        "fingertip",        # index
        "fingertip_2",      # middle
        "fingertip_3",      # ring
        "thumb_fingertip",  # thumb
    ]


    def __init__(self, *args, robot_uids="my_leap_hand", **kwargs):
        self.cube_half_size = 0.03
        self.cube_pos       = np.array([-0.4, 0.0, self.cube_half_size])
        self.goal_height    = 0.15    # lift target above table
        self.goal_thresh    = 0.05    # success radius

        # Side-grasp spawn: hand orbits the cube at this radius & height
        self.orbit_radius   = 0.03
        self.hand_spawn_z   = 0.1   # keep palm roughly at cube mid-height

        # Base orientation that makes fingers point inward (toward cube)
        # when yaw is 0 — tune this to match your URDF's natural grasp axis
        self._base_q = np.array([0.7071068, -0.7071068, 0.0, 0.0])

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    def _load_agent(self, options: dict):
        # Initial pose — will be overwritten in _initialize_episode
        super()._load_agent(
            options,
            sapien.Pose(p=[-0.4 + self.orbit_radius, 0.0, self.hand_spawn_z],
                        q=self._base_q.tolist())
        )

    # ------------------------------------------------------------------
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=self.cube_pos.tolist()),
        )

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.4],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=[self.cube_pos[0], self.cube_pos[1], self.goal_height]
            ),
        )

    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ── Fixed cube pose ──────────────────────────────────────
            cube_xyz = torch.tensor(
                self.cube_pos, dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(b, -1).clone()
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

            # ── Fixed goal pose ──────────────────────────────────────
            goal_xyz = torch.tensor(
                [self.cube_pos[0], self.cube_pos[1], self.goal_height],
                dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(b, -1).clone()
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

            # ── Hand spawns in a random orbit around the cube ────────
            # Random yaw angle for each env
            theta = torch.rand(b, device=self.device) * 2.0 * torch.pi

            hand_x = cube_xyz[:, 0] + self.orbit_radius * torch.cos(theta)
            hand_y = cube_xyz[:, 1] + self.orbit_radius * torch.sin(theta)
            hand_z = torch.full((b,), self.hand_spawn_z, device=self.device)
            hand_pos = torch.stack([hand_x, hand_y, hand_z], dim=1)

            # Yaw quaternion: rotate so palm faces the cube
            # direction from hand toward cube
            dx = cube_xyz[:, 0] - hand_x   # (b,)
            dy = cube_xyz[:, 1] - hand_y   # (b,)
            yaw = torch.atan2(dy, dx)       # (b,)

            # Yaw-only quaternion (rotation around world Z)
            half_yaw = yaw * 0.5
            yaw_q = torch.stack([
                torch.cos(half_yaw),        # w
                torch.zeros_like(half_yaw), # x
                torch.zeros_like(half_yaw), # y
                torch.sin(half_yaw),        # z
            ], dim=1)  # (b, 4)

            # Base orientation (WXYZ)
            base_q = torch.tensor(
                self._base_q, dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(b, -1)  # (b, 4)

            # Compose: final = yaw_q * base_q  (Hamilton product)
            hand_quat = self._quat_mul(yaw_q, base_q)  # (b, 4)

            self.agent.robot.set_pose(
                Pose.create_from_pq(p=hand_pos, q=hand_quat)
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two (b,4) WXYZ quaternion tensors."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=1)

    # ------------------------------------------------------------------
    def _get_fingertip_links(self):
        links = []
        for name in self.FINGERTIP_NAMES:
            link = self.agent.robot.find_link_by_name(name)
            if link is not None:
                links.append(link)
        return links

    def _get_obs_extra(self, info: dict):
        # Mean fingertip position — useful observation for side grasp
        fingertip_links = self._get_fingertip_links()
        if fingertip_links:
            fingertip_pos = torch.stack(
                [l.pose.p for l in fingertip_links], dim=1
            ).mean(dim=1)  # (num_envs, 3)
        else:
            fingertip_pos = self.agent.tcp.pose.p

        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            fingertip_pos=fingertip_pos,
            obj_pose=self.cube.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )

    # ------------------------------------------------------------------
    def evaluate(self):
        obj_to_goal_dist = torch.linalg.norm(
            self.cube.pose.p - self.goal_site.pose.p, dim=1
        )
        success = obj_to_goal_dist < self.goal_thresh
        return {"success": success, "obj_to_goal_dist": obj_to_goal_dist}

    # ------------------------------------------------------------------
    def _is_grasping(self, obj) -> torch.Tensor:
        """
        Returns float (num_envs,): 1.0 if total fingertip contact force > threshold.
        Also returns per-finger contact count for the fingertip reward.
        """
        fingertip_links = self._get_fingertip_links()
        total_force    = torch.zeros(self.num_envs, device=self.device)
        fingers_in_contact = torch.zeros(self.num_envs, device=self.device)

        for link in fingertip_links:
            try:
                forces = self.scene.get_pairwise_contact_forces(link, obj)
                force_mag = torch.linalg.norm(forces, dim=-1)
                total_force += force_mag
                fingers_in_contact += (force_mag > 0.1).float()
            except Exception:
                pass

        is_grasped = (total_force > 0.5).float()
        return is_grasped, fingers_in_contact

    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cube_pos = self.cube.pose.p   # (num_envs, 3)

        # ── 1. Fingertip approach reward (better than palm for side grasp) ──
        fingertip_links = self._get_fingertip_links()
        if fingertip_links:
            # mean distance of all fingertips to cube center
            ft_positions = torch.stack(
                [l.pose.p for l in fingertip_links], dim=1
            )  # (num_envs, 4, 3)
            ft_to_cube = torch.linalg.norm(
                ft_positions - cube_pos.unsqueeze(1), dim=-1
            )  # (num_envs, 4)
            mean_ft_dist = ft_to_cube.mean(dim=1)  # (num_envs,)
        else:
            mean_ft_dist = torch.linalg.norm(
                cube_pos - self.agent.tcp.pose.p, dim=1
            )

        reach_reward = 1.0 - torch.tanh(3.0 * mean_ft_dist) 

        # ── 2. Finger spread reward ─────────────────────────────────────────
        # Encourage fingers to surround the cube (low variance in angle)
        # Reward more fingers being in contact
        is_grasped, fingers_in_contact = self._is_grasping(self.cube)
        contact_reward = fingers_in_contact * 0.3   # up to 1.2 for all 4 fingers

        # ── 3. Grasp quality bonus ───────────────────────────────────────────
        grasp_reward = is_grasped * 2.0

        # ── 4. Lift reward (only when grasped) ───────────────────────────────
        obj_to_goal_dist = info["obj_to_goal_dist"]
        lift_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_dist)) * is_grasped

        # ── 5. Keep cube from being knocked away (horizontal displacement) ───
        cube_xy = cube_pos[:, :2]
        target_xy = torch.tensor(
            self.cube_pos[:2], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        cube_displaced = torch.linalg.norm(cube_xy - target_xy, dim=1)
        # small penalty if cube slides sideways (means hand pushed it)
        stability_penalty = -0.5 * torch.clamp(cube_displaced - 0.05, min=0.0)

        # ── 6. Action smoothness penalty ─────────────────────────────────────
        action_penalty = -0.01 * torch.linalg.norm(action, dim=-1)

        reward = (
            reach_reward
            + contact_reward
            + grasp_reward
            + lift_reward
            + stability_penalty
            + action_penalty
        )

        # ── 7. Success bonus ─────────────────────────────────────────────────
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10.0


# =============================================================================
# 3. LAUNCHER
# =============================================================================
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "PickCubeLeap-v1",
        render_mode="human",
        reward_mode="dense",
    )

    obs, _ = env.reset()
    uw = env.unwrapped
    print("Environment loaded!")
    print(f"  Cube position : {uw.cube_pos}")
    print(f"  Goal height   : {uw.goal_height} m")
    print(f"  Orbit radius  : {uw.orbit_radius} m")
    print(f"  Action space  : {env.action_space.shape}")
    print(f"  Obs space     : {env.observation_space}")

    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"  Step {step:4d} | success={info['success'].any().item()}")
            obs, _ = env.reset()

    env.close()
