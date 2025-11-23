# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = []
            for key in env.rew.keys():
                if key != "death_cost":
                    # Remove common suffixes to get the base reward name
                    base_key = key.replace("_reward_scale", "").replace("_penalty_scale", "")
                    keys.append(base_key)
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py."""

        # Recalculate pose relative to current target gate
        drone_pose = self.env._robot.data.root_link_state_w[:, :3]
        self.env._pose_drone_wrt_gate, _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp, :3],
            self.env._waypoints_quat[self.env._idx_wp, :],
            drone_pose
        )

        # Target position: 2.0m beyond the gate center (negative X because we fly +X -> -X)
        target_pos_gate_frame = torch.tensor([-2.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        
        # Current distance to target
        dist_to_target = torch.norm(self.env._pose_drone_wrt_gate - target_pos_gate_frame, dim=1)
        
        # Initialize prev_dist if not exists (handle first step)
        if not hasattr(self.env, '_prev_dist_to_target'):
            self.env._prev_dist_to_target = dist_to_target.clone()

        # 1. Progress Reward: Change in distance to target
        # Positive if getting closer, negative if moving away
        progress_reward = (self.env._prev_dist_to_target - dist_to_target)
        
        # Update previous distance
        self.env._prev_dist_to_target = dist_to_target.clone()

        # 1.5 Stagnation penalty: if we are near the target but not making progress
        # (especially on later gates), add a small penalty to discourage hovering.
        near_target = dist_to_target < 0.6
        low_progress = torch.abs(progress_reward) < 1e-3
        later_gates = self.env._idx_wp >= 4  # only from 5th gate onward
        stagnation_mask = near_target & low_progress & later_gates
        stagnation_penalty = stagnation_mask.float()

        # 2. Distance Reward: Exponential penalty for being far from target
        distance_reward = torch.exp(-0.5 * dist_to_target)

        # 2.5 Centering Reward: Stronger penalty for lateral/vertical deviation from gate axis
        # y and z in gate frame
        y_gate = self.env._pose_drone_wrt_gate[:, 1]
        z_gate = self.env._pose_drone_wrt_gate[:, 2]
        dist_from_center = torch.sqrt(y_gate**2 + z_gate**2)
        centering_reward = torch.exp(-2.0 * dist_from_center)

        # 3. Gate Pass Reward
        # Detect gate crossing: x crosses from negative to positive (behind -> in front)
        current_x = self.env._pose_drone_wrt_gate[:, 0]
        prev_x = self.env._prev_x_drone_wrt_gate
        
        gate_size = self.env._gate_model_cfg_data.gate_side
        within_gate_y = torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < gate_size / 2.0
        within_gate_z = torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < gate_size / 2.0
        within_gate_bounds = within_gate_y & within_gate_z
        
        crossed_plane = (prev_x > 0) & (current_x <= 0)
        gate_passed = crossed_plane & within_gate_bounds
        
        gate_pass_reward = gate_passed.float()
        
        # Update previous x
        self.env._prev_x_drone_wrt_gate = current_x.clone()

        # Handle gate transition
        ids_gate_passed = torch.where(gate_passed)[0]
        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1
            
            # Update desired pos
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
            
            # Recalculate pose for new gate
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                drone_pose[ids_gate_passed]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = self.env._pose_drone_wrt_gate[ids_gate_passed, 0]
            
            # Reset distance tracking for new gate
            new_dist = torch.norm(self.env._pose_drone_wrt_gate[ids_gate_passed] - target_pos_gate_frame[ids_gate_passed], dim=1)
            self.env._prev_dist_to_target[ids_gate_passed] = new_dist

        # 4. Alignment Reward
        # Dot product of drone forward vector and vector to target
        # Drone forward vector in world frame
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_rot_mat = matrix_from_quat(drone_quat_w)
        drone_forward_w = drone_rot_mat[:, :, 0] # X-axis is forward
        
        # Vector to target in world frame
        target_pos_w = self.env._waypoints[self.env._idx_wp, :3] # Approximate target as gate center for alignment
        # Better: Transform target_pos_gate_frame to world
        # But for alignment, pointing to gate center is good enough
        vec_to_target_w = target_pos_w - drone_pose
        vec_to_target_w_norm = vec_to_target_w / (torch.norm(vec_to_target_w, dim=1, keepdim=True) + 1e-6)
        
        alignment_reward = torch.sum(drone_forward_w * vec_to_target_w_norm, dim=1)

        # 5. Velocity Reward
        # Project velocity onto vector to target
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w
        vel_proj = torch.sum(drone_vel_w * vec_to_target_w_norm, dim=1)
        velocity_reward = torch.clamp(vel_proj, min=0.0)

        # 6. Penalties
        # Crash
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int() # Ignore initial contacts
        self.env._crashed = self.env._crashed + crashed * mask
        
        # Low altitude
        height = self.env._robot.data.root_link_pos_w[:, 2]
        low_altitude_penalty = torch.clamp(0.2 - height, min=0.0) # Penalize if below 0.2m
        
        # Angular velocity penalty (smoothness)
        ang_vel = self.env._robot.data.root_ang_vel_b
        ang_vel_penalty = torch.sum(torch.abs(ang_vel), dim=1)

        if self.cfg.is_train:
            rewards = {
                "gate_pass": gate_pass_reward * self.env.rew['gate_pass_reward_scale'],
                "progress": progress_reward * self.env.rew['progress_reward_scale'],
                "stagnation": stagnation_penalty * self.env.rew['stagnation_penalty_scale'],
                "distance": distance_reward * self.env.rew['distance_reward_scale'],
                "centering": centering_reward * self.env.rew['centering_reward_scale'],
                "alignment": alignment_reward * self.env.rew['alignment_reward_scale'],
                "velocity": velocity_reward * self.env.rew['velocity_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
                "low_altitude": low_altitude_penalty * self.env.rew['low_altitude_penalty_scale'],
                "ang_vel": ang_vel_penalty * self.env.rew['ang_vel_penalty_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations."""

        # Basic drone states in body frame
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        
        # Orientation: Rotation matrix (flattened) to avoid Euler singularities
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_rot_mat = matrix_from_quat(drone_quat_w).reshape(self.num_envs, 9)
        
        # Relative position to gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate
        
        # Relative velocity in gate frame
        current_gate_idx = self.env._idx_wp
        current_gate_quat = self.env._waypoints_quat[current_gate_idx, :]
        rotation_matrices = matrix_from_quat(current_gate_quat)
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w
        drone_vel_gate_frame = torch.bmm(
            rotation_matrices.transpose(1, 2),
            drone_vel_w.unsqueeze(-1)
        ).squeeze(-1)
        
        # Next gate relative position
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        
        next_gate_rel_pos, _ = subtract_frame_transforms(
            self.env._waypoints[current_gate_idx, :3],
            current_gate_quat,
            next_gate_pos_w
        )
        
        # Previous actions
        prev_actions = self.env._previous_actions
        
        # Vector to next gate in body frame (Crucial for anticipation)
        # next_gate_pos_w is already calculated
        drone_pos_w = self.env._robot.data.root_link_pos_w
        vec_to_next_gate_w = next_gate_pos_w - drone_pos_w
        
        # Rotate to body frame (need 3x3 matrices here, keep separate from flattened version)
        drone_rot_mat_3d = matrix_from_quat(drone_quat_w)
        vec_to_next_gate_b = torch.bmm(
            drone_rot_mat_3d.transpose(1, 2),
            vec_to_next_gate_w.unsqueeze(-1)
        ).squeeze(-1)
        
        obs = torch.cat(
            [
                drone_lin_vel_b,           # (3)
                drone_ang_vel_b,           # (3)
                drone_rot_mat,             # (9)
                drone_pos_gate_frame,      # (3)
                drone_vel_gate_frame,      # (3)
                next_gate_rel_pos,         # (3)
                vec_to_next_gate_b,        # (3) [NEW]
                prev_actions,              # (4)
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # Initialize drone states
        num_waypoints = self.env._waypoints.shape[0]
        
        # Curriculum: Start from random gates
        # 50% chance to start from gate 0, 50% random gate
        random_start_mask = torch.rand(n_reset, device=self.device) < 0.5
        random_waypoint_indices = torch.randint(0, num_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
        waypoint_indices = torch.where(random_start_mask, random_waypoint_indices, torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype))

        # Get gate poses
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]
        theta = self.env._waypoints[waypoint_indices][:, -1]

        # Randomize position behind gate (Cone/Box)
        # x: -1.5m to -2.5m behind gate
        x_local = torch.empty(n_reset, device=self.device).uniform_(-2.5, -1.5)
        # y: +/- 0.5m lateral
        y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
        # z: +/- 0.25m vertical
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.25, 0.25)

        # Rotate to world frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        
        initial_x = x0_wp - x_rot # Invert direction based on user feedback
        initial_y = y0_wp - y_rot
        initial_z = torch.clamp(z_wp + z_local, min=0.5) # Increase min altitude

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Orientation: Point towards gate with noise
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3) # +/- ~17 deg
        roll_noise = torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1)
        pitch_noise = torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1)
        
        quat = quat_from_euler_xyz(roll_noise, pitch_noise, initial_yaw + yaw_noise)
        default_root_state[:, 3:7] = quat

        # Initial velocity: Small forward velocity (avoid blasting too fast)
        # 0 to 0.5 m/s forward
        speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 0.5)
        vel_x_local = speed
        vel_y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
        vel_z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
        
        # Rotate velocity to world
        vel_x_w = cos_theta * vel_x_local - sin_theta * vel_y_local
        vel_y_w = sin_theta * vel_x_local + cos_theta * vel_y_local
        
        default_root_state[:, 7] = vel_x_w
        default_root_state[:, 8] = vel_y_w
        default_root_state[:, 9] = vel_z_local

        # Handle play mode (eval)
        if not self.cfg.is_train:
            # Start at gate 0, 2m behind
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            x0_wp = self.env._waypoints[waypoint_indices][:, 0]
            y0_wp = self.env._waypoints[waypoint_indices][:, 1]
            z_wp = self.env._waypoints[waypoint_indices][:, 2]
            theta = self.env._waypoints[waypoint_indices][:, -1]
            
            x_local = -2.0
            y_local = 0.0
            z_local = 0.0
            
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            
            initial_x = x0_wp - (cos_theta * x_local - sin_theta * y_local) # Invert here too
            initial_y = y0_wp - (sin_theta * x_local + cos_theta * y_local)
            initial_z = z_wp # Start at gate center height
            
            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            
            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            quat = quat_from_euler_xyz(torch.zeros_like(initial_yaw), torch.zeros_like(initial_yaw), initial_yaw)
            default_root_state[:, 3:7] = quat
            default_root_state[:, 7:] = 0.0 # Stationary start

        # Set waypoint indices
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0
        self.env._crashed[env_ids] = 0

        # Recalculate pose relative to gate
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0]
        
        # Initialize prev_dist_to_target for rewards
        target_pos_gate_frame = torch.tensor([-2.0, 0.0, 0.0], device=self.device).expand(n_reset, 3)
        dist_to_target = torch.norm(self.env._pose_drone_wrt_gate[env_ids] - target_pos_gate_frame, dim=1)
        
        if not hasattr(self.env, '_prev_dist_to_target'):
             self.env._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.env._prev_dist_to_target[env_ids] = dist_to_target