# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            observations,
            critic_observations,
            sampled_actions,
            value_targets,
            advantage_estimates,
            discounted_returns,
            prev_log_probs,
            prev_mean_actions,
            prev_action_stds,
            hidden_states,
            episode_masks,
            _,  # rnd_state_batch - not used anymore
        ) in generator:
            # Flatten the tensors we need for loss computation
            actions = sampled_actions.contiguous().view(-1, sampled_actions.shape[-1])
            old_log_probs = prev_log_probs.contiguous().view(-1)
            target_values = value_targets.contiguous().view(-1, 1)
            returns = discounted_returns.contiguous().view(-1, 1)
            advantages = advantage_estimates.contiguous().view(-1)
            old_mu = prev_mean_actions.contiguous().view(-1, prev_mean_actions.shape[-1])
            old_sigma = prev_action_stds.contiguous().view(-1, prev_action_stds.shape[-1])

            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # Recompute policy distribution and value estimates
            if self.actor_critic.is_recurrent:
                act_hidden, crit_hidden = hidden_states
                self.actor_critic.act(observations, masks=episode_masks, hidden_states=act_hidden)
                actions_log_prob = self.actor_critic.get_actions_log_prob(actions)
                values = self.actor_critic.evaluate(
                    critic_observations, masks=episode_masks, hidden_states=crit_hidden
                )
            else:
                self.actor_critic.update_distribution(observations)
                actions_log_prob = self.actor_critic.get_actions_log_prob(actions)
                values = self.actor_critic.evaluate(critic_observations)

            values = values.contiguous().view(-1, 1)
            log_probs = actions_log_prob.contiguous().view(-1)
            entropy = self.actor_critic.entropy
            entropy_loss = entropy.mean()

            # Policy loss (clipped surrogate objective)
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate = ratio * advantages
            surrogate_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                values_clipped = target_values + (values - target_values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - returns).pow(2), (values_clipped - returns).pow(2)).mean()
            else:
                value_loss = (returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

            # Adaptive learning-rate scheduling (if requested)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    new_mu = self.actor_critic.action_mean.contiguous().view(-1, prev_mean_actions.shape[-1])
                    new_sigma = self.actor_critic.action_std.contiguous().view(-1, prev_action_stds.shape[-1])
                    kl = torch.sum(
                        torch.log(new_sigma / (old_sigma + 1e-8))
                        + (old_sigma.pow(2) + (old_mu - new_mu).pow(2)) / (2.0 * new_sigma.pow(2))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            # Apply gradients
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate statistics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy
