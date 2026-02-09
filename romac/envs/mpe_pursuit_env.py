import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def clamp(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


@dataclass
class AgentState:
    position: np.ndarray
    velocity: np.ndarray


class MPEPursuitEnv:
    """Multi-UAV pursuit environment with hunters, blockers, and a target."""

    def __init__(self, config: Dict):
        self.map_size = float(config["map_size"])
        self.max_steps = int(config["max_steps"])
        self.capture_distance = float(config["capture_distance"])
        self.capture_hold_steps = int(config["capture_hold_steps"])
        self.hunter_speed = float(config["hunter_speed"])
        self.blocker_speed = float(config["blocker_speed"])
        self.target_speed = float(config["target_speed"])
        self.hunter_perception = float(config["hunter_perception"])
        self.blocker_perception = float(config["blocker_perception"])
        self.target_perception = float(config["target_perception"])
        self.num_hunters = int(config["num_hunters"])
        self.num_blockers = int(config["num_blockers"])
        self.num_targets = int(config["num_targets"])
        self.max_hunters = int(config["max_hunters"])
        self.max_blockers = int(config["max_blockers"])
        if self.num_targets != 1:
            raise ValueError("Current implementation expects one target.")
        if self.num_hunters > self.max_hunters or self.num_blockers > self.max_blockers:
            raise ValueError("num_hunters/blockers exceed max limits.")

        self.rng = np.random.default_rng()
        self.step_count = 0
        self.capture_counter = 0
        self.hunters: List[AgentState] = []
        self.blockers: List[AgentState] = []
        self.targets: List[AgentState] = []

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @property
    def agents(self) -> List[str]:
        hunters = [f"hunter_{i}" for i in range(self.num_hunters)]
        blockers = [f"blocker_{i}" for i in range(self.num_blockers)]
        targets = [f"target_{i}" for i in range(self.num_targets)]
        return hunters + blockers + targets

    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.capture_counter = 0
        self.hunters = [self._random_state(self.hunter_speed) for _ in range(self.num_hunters)]
        self.blockers = [self._random_state(self.blocker_speed) for _ in range(self.num_blockers)]
        self.targets = [self._random_state(self.target_speed) for _ in range(self.num_targets)]
        return self._get_obs()

    def _random_state(self, max_speed: float) -> AgentState:
        pos = self.rng.uniform(-self.map_size, self.map_size, size=2)
        vel = self.rng.uniform(-max_speed, max_speed, size=2)
        return AgentState(position=pos, velocity=vel)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        for idx in range(self.num_hunters):
            obs[f"hunter_{idx}"] = self._agent_obs(
                self.hunters[idx],
                self.hunter_perception,
                self.max_hunters,
                self.max_blockers,
            )
        for idx in range(self.num_blockers):
            obs[f"blocker_{idx}"] = self._agent_obs(
                self.blockers[idx],
                self.blocker_perception,
                self.max_hunters,
                self.max_blockers,
            )
        for idx in range(self.num_targets):
            obs[f"target_{idx}"] = self._target_obs(self.targets[idx])
        return obs

    def _agent_obs(
        self,
        agent: AgentState,
        perception: float,
        max_hunters: int,
        max_blockers: int,
    ) -> np.ndarray:
        target = self.targets[0]
        rel_target = self._relative(agent.position, target.position, perception)
        hunter_rel = self._padded_relations(agent.position, self.hunters, max_hunters, perception)
        blocker_rel = self._padded_relations(agent.position, self.blockers, max_blockers, perception)
        return np.concatenate([agent.position, agent.velocity, rel_target, hunter_rel, blocker_rel])

    def _target_obs(self, target: AgentState) -> np.ndarray:
        hunter_rel = self._padded_relations(
            target.position, self.hunters, self.max_hunters, self.target_perception
        )
        blocker_rel = self._padded_relations(
            target.position, self.blockers, self.max_blockers, self.target_perception
        )
        return np.concatenate([target.position, target.velocity, hunter_rel, blocker_rel])

    def _relative(self, src: np.ndarray, dst: np.ndarray, perception: float) -> np.ndarray:
        diff = dst - src
        distance = np.linalg.norm(diff)
        if distance <= perception:
            return diff
        return np.zeros(2, dtype=np.float32)

    def _padded_relations(
        self, src: np.ndarray, group: List[AgentState], max_size: int, perception: float
    ) -> np.ndarray:
        rels = []
        for agent in group:
            diff = self._relative(src, agent.position, perception)
            rels.append(diff)
        while len(rels) < max_size:
            rels.append(np.zeros(2, dtype=np.float32))
        return np.concatenate(rels[:max_size])

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        self.step_count += 1
        self._apply_actions(actions)
        rewards = self._compute_rewards()
        dones = {agent: False for agent in self.agents}
        infos: Dict[str, Dict] = {agent: {} for agent in self.agents}

        if self._check_capture():
            for agent in dones:
                dones[agent] = True
            infos["capture"] = {"success": True}
        if self.step_count >= self.max_steps:
            for agent in dones:
                dones[agent] = True
        return self._get_obs(), rewards, dones, infos

    def _apply_actions(self, actions: Dict[str, np.ndarray]) -> None:
        for idx in range(self.num_hunters):
            action = actions.get(f"hunter_{idx}", np.zeros(2))
            self._move(self.hunters[idx], action, self.hunter_speed)
        for idx in range(self.num_blockers):
            action = actions.get(f"blocker_{idx}", np.zeros(2))
            self._move(self.blockers[idx], action, self.blocker_speed)
        for idx in range(self.num_targets):
            action = actions.get(f"target_{idx}", np.zeros(2))
            self._move(self.targets[idx], action, self.target_speed)

    def _move(self, agent: AgentState, action: np.ndarray, max_speed: float) -> None:
        accel = clamp(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        agent.velocity = clamp(agent.velocity + accel * 0.1, -max_speed, max_speed)
        agent.position = clamp(agent.position + agent.velocity * 0.1, -self.map_size, self.map_size)

    def _compute_rewards(self) -> Dict[str, float]:
        target = self.targets[0]
        hunter_distances = [
            np.linalg.norm(h.position - target.position) for h in self.hunters
        ]
        min_hunter_distance = min(hunter_distances) if hunter_distances else 0.0
        capture_reward = 10.0 if self._check_capture(peek=True) else 0.0
        hunter_reward = capture_reward - 0.1 * min_hunter_distance
        blocker_reward = capture_reward - 0.05 * min_hunter_distance
        target_reward = 0.1 * min_hunter_distance - capture_reward
        rewards: Dict[str, float] = {}
        for idx in range(self.num_hunters):
            rewards[f"hunter_{idx}"] = hunter_reward
        for idx in range(self.num_blockers):
            rewards[f"blocker_{idx}"] = blocker_reward
        for idx in range(self.num_targets):
            rewards[f"target_{idx}"] = target_reward
        return rewards

    def _check_capture(self, peek: bool = False) -> bool:
        target = self.targets[0]
        distances = [np.linalg.norm(h.position - target.position) for h in self.hunters]
        if distances and min(distances) <= self.capture_distance:
            if peek:
                return True
            self.capture_counter += 1
        else:
            self.capture_counter = 0
        return self.capture_counter >= self.capture_hold_steps

    def target_allocation(self, targets: List[np.ndarray]) -> List[int]:
        """Assign trackers to targets (placeholder for future multi-target extension)."""
        return [0 for _ in targets]

    def reassign_trackers(
        self, current_assignments: Dict[str, int], new_target: int
    ) -> Dict[str, int]:
        """Reassign trackers to a different target (placeholder for dynamic re-tasking)."""
        return {agent: new_target for agent in current_assignments}

    def observation_space_dim(self) -> Dict[str, int]:
        hunter_dim = 2 + 2 + 2 + self.max_hunters * 2 + self.max_blockers * 2
        blocker_dim = hunter_dim
        target_dim = 2 + 2 + self.max_hunters * 2 + self.max_blockers * 2
        return {"hunter": hunter_dim, "blocker": blocker_dim, "target": target_dim}
