import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

if __package__ is None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from romac.algorithms import MAPPOTrainer
from romac.algorithms.mappo import PPOTrajectory
from romac.envs import MPEPursuitEnv
from romac.utils import load_config, prepare_run_dir


def main() -> None:
    config_path = Path(__file__).resolve().with_name("config.yaml")
    config = load_config(str(config_path))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MPEPursuitEnv(config["env"])
    env.seed(seed)
    obs_dims = env.observation_space_dim()

    run_dir = prepare_run_dir(config["output_dir"], config["run_name"])
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    writer = SummaryWriter(log_dir=run_dir / "tensorboard")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = MAPPOTrainer(config["training"], obs_dims, config["model"]["action_dim"], device)

    total_episodes = int(config["training"]["total_episodes"])
    log_interval = int(config["training"]["log_interval"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])

    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        trajectories = {
            "hunter": [],
            "blocker": [],
            "target": [],
        }
        episode_rewards = {"hunter": 0.0, "blocker": 0.0, "target": 0.0}

        for _ in range(env.max_steps):
            actions, log_probs, values = trainer.collect_actions(obs)
            next_obs, rewards, dones, _ = env.step(actions)

            for agent_id, obs_vec in obs.items():
                role = agent_id.split("_")[0]
                trajectories[role].append(
                    (
                        obs_vec,
                        actions[agent_id],
                        log_probs[agent_id],
                        rewards[agent_id],
                        dones[agent_id],
                        values[agent_id],
                    )
                )
                episode_rewards[role] += rewards[agent_id]

            obs = next_obs
            if all(dones.values()):
                break

        ppo_trajectories = {}
        for role, transitions in trajectories.items():
            obs_list, action_list, log_list, reward_list, done_list, value_list = zip(*transitions)
            ppo_trajectories[role] = PPOTrajectory(
                obs=list(obs_list),
                actions=list(action_list),
                log_probs=list(log_list),
                rewards=list(reward_list),
                dones=list(done_list),
                values=list(value_list),
            )

        metrics = trainer.update(ppo_trajectories)

        if episode % log_interval == 0:
            for role, reward in episode_rewards.items():
                writer.add_scalar(f"reward/{role}", reward, episode)
            for metric_name, value in metrics.items():
                writer.add_scalar(metric_name, value, episode)

        if episode % checkpoint_interval == 0:
            for role, agent in trainer.role_agents.items():
                torch.save(
                    agent.actor.state_dict(),
                    run_dir / "checkpoints" / f"{role}_actor_{episode}.pt",
                )
                torch.save(
                    agent.critic.state_dict(),
                    run_dir / "checkpoints" / f"{role}_critic_{episode}.pt",
                )

    for role, agent in trainer.role_agents.items():
        torch.save(agent.actor.state_dict(), run_dir / "models" / f"{role}_actor.pt")
        torch.save(agent.critic.state_dict(), run_dir / "models" / f"{role}_critic.pt")

    writer.close()


if __name__ == "__main__":
    main()
