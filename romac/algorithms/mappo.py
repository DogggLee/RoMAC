from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(obs)
        mean = self.mean(hidden)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class PPOTrajectory:
    obs: List[np.ndarray]
    actions: List[np.ndarray]
    log_probs: List[float]
    rewards: List[float]
    dones: List[bool]
    values: List[float]


class MAPPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, device: str):
        self.actor = Actor(obs_dim, action_dim, hidden_size).to(device)
        self.critic = Critic(obs_dim, hidden_size).to(device)
        self.device = device

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action, log_prob = self.actor.act(obs_tensor)
        value = self.critic(obs_tensor)
        return action.detach().cpu().numpy(), log_prob.item(), value.item()


class MAPPOTrainer:
    def __init__(
        self,
        config: Dict,
        model_config: Dict,
        obs_dims: Dict[str, int],
        action_dim: int,
        device: str,
    ):
        hidden_size = int(model_config["hidden_size"])
        self.role_agents = {
            "hunter": MAPPOAgent(obs_dims["hunter"], action_dim, hidden_size, device),
            "blocker": MAPPOAgent(obs_dims["blocker"], action_dim, hidden_size, device),
            "target": MAPPOAgent(obs_dims["target"], action_dim, hidden_size, device),
        }
        self.device = device
        self.gamma = float(config["gamma"])
        self.gae_lambda = float(config["gae_lambda"])
        self.ppo_epochs = int(config["ppo_epochs"])
        self.clip_ratio = float(config["clip_ratio"])
        self.entropy_coef = float(config["entropy_coef"])
        self.value_coef = float(config["value_coef"])
        self.max_grad_norm = float(config["max_grad_norm"])
        self.batch_size = int(config["batch_size"])
        self.actor_optims = {}
        self.critic_optims = {}
        for role, agent in self.role_agents.items():
            self.actor_optims[role] = torch.optim.Adam(
                agent.actor.parameters(), lr=float(config["actor_lr"])
            )
            self.critic_optims[role] = torch.optim.Adam(
                agent.critic.parameters(), lr=float(config["critic_lr"])
            )

    def collect_actions(self, obs: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict]:
        actions = {}
        log_probs = {}
        values = {}
        for agent_id, obs_vec in obs.items():
            role = agent_id.split("_")[0]
            action, log_prob, value = self.role_agents[role].act(obs_vec)
            actions[agent_id] = action
            log_probs[agent_id] = log_prob
            values[agent_id] = value
        return actions, log_probs, values

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        advantages = []
        gae = 0.0
        values = values + [0.0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def update(self, trajectories: Dict[str, PPOTrajectory]) -> Dict[str, float]:
        metrics = {}
        for role, traj in trajectories.items():
            obs = torch.tensor(np.array(traj.obs), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(traj.actions), dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(np.array(traj.log_probs), dtype=torch.float32, device=self.device)
            values = torch.tensor(np.array(traj.values), dtype=torch.float32, device=self.device)
            advantages, returns = self.compute_gae(traj.rewards, traj.values, traj.dones)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor = self.role_agents[role].actor
            critic = self.role_agents[role].critic
            actor_optim = self.actor_optims[role]
            critic_optim = self.critic_optims[role]

            dataset_size = obs.size(0)
            for _ in range(self.ppo_epochs):
                indices = torch.randperm(dataset_size)
                for start in range(0, dataset_size, self.batch_size):
                    batch_idx = indices[start : start + self.batch_size]
                    batch_obs = obs[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_adv = advantages[batch_idx]
                    batch_returns = returns[batch_idx]

                    mean, log_std = actor(batch_obs)
                    dist = Normal(mean, log_std.exp())
                    log_probs = dist.log_prob(batch_actions).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()

                    ratio = (log_probs - batch_old_log_probs).exp()
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_adv
                    actor_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * entropy)

                    value_pred = critic(batch_obs)
                    critic_loss = ((batch_returns - value_pred) ** 2).mean()

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    actor_optim.step()

                    critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                    critic_optim.step()

            metrics[f"{role}_actor_loss"] = actor_loss.item()
            metrics[f"{role}_critic_loss"] = critic_loss.item()
        return metrics
