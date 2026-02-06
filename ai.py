"""Adaptive platformer AI scaffold.

This module implements a practical, modular architecture for building a speedrun-
focused agent that can adapt across different platformers.

It includes:
- hierarchical decision making (high-level planner + low-level controller),
- curriculum scheduling,
- hybrid state representation (vision + structured features),
- lightweight physics/world-model reasoning,
- few-shot style adaptation memory,
- explicit speed-first reward shaping.

The implementation intentionally avoids heavy dependencies so the design can be
integrated into existing training stacks (PPO/SAC/model-based RL) later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple


class Action(str, Enum):
    """Discrete frame action vocabulary for a platformer controller."""

    IDLE = "idle"
    LEFT = "left"
    RIGHT = "right"
    JUMP = "jump"
    LEFT_JUMP = "left_jump"
    RIGHT_JUMP = "right_jump"


class HighLevelGoal(str, Enum):
    """Strategic goals for the high-level planner."""

    REACH_FINISH = "reach_finish"
    RECOVER = "recover"
    ATTEMPT_SHORTCUT = "attempt_shortcut"
    SAFE_PROGRESS = "safe_progress"


@dataclass
class Observation:
    """Input observation from a game wrapper.

    `vision` can be any game observation payload (frame, latent feature, etc.).
    `features` should carry structured values when available.
    """

    vision: Any
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class StateEmbedding:
    """Unified game-agnostic state vector."""

    values: List[float]


@dataclass
class Transition:
    """Single transition for online adaptation/world-model updates."""

    state: StateEmbedding
    action: Action
    reward: float
    next_state: StateEmbedding
    done: bool


class VisionEncoder(Protocol):
    def encode(self, vision: Any) -> List[float]:
        ...


class BasicVisionEncoder:
    """Fallback encoder that maps generic numeric inputs into fixed-size vectors."""

    def __init__(self, output_dim: int = 8) -> None:
        self.output_dim = output_dim

    def encode(self, vision: Any) -> List[float]:
        if isinstance(vision, (int, float)):
            base = [float(vision)]
        elif isinstance(vision, (list, tuple)) and vision:
            numeric = [float(x) for x in vision if isinstance(x, (int, float))]
            base = numeric if numeric else [0.0]
        else:
            base = [0.0]

        out = [0.0] * self.output_dim
        for i in range(self.output_dim):
            out[i] = base[i % len(base)] / (1.0 + i)
        return out


class PerceptionModule:
    """Builds a hybrid representation from vision + structured features."""

    def __init__(self, encoder: Optional[VisionEncoder] = None) -> None:
        self.encoder = encoder or BasicVisionEncoder()
        self.feature_keys: List[str] = []

    def fit_feature_schema(self, feature_keys: Iterable[str]) -> None:
        self.feature_keys = sorted(set(feature_keys))

    def embed(self, obs: Observation) -> StateEmbedding:
        vision_latent = self.encoder.encode(obs.vision)
        if not self.feature_keys:
            self.fit_feature_schema(obs.features.keys())

        structured = [float(obs.features.get(k, 0.0)) for k in self.feature_keys]
        return StateEmbedding(values=vision_latent + structured)


@dataclass
class PhysicsEstimate:
    gravity: float = 0.35
    jump_impulse: float = 5.0
    horizontal_drag: float = 0.03
    uncertainty: float = 1.0


class WorldModel:
    """Simple physics-aware predictor with online parameter adaptation."""

    def __init__(self) -> None:
        self.physics = PhysicsEstimate()

    def predict_next(self, state: StateEmbedding, action: Action) -> StateEmbedding:
        # Convention: [x, y, vx, vy, ...] when available.
        v = list(state.values)
        while len(v) < 4:
            v.append(0.0)

        x, y, vx, vy = v[0], v[1], v[2], v[3]

        if action in (Action.LEFT, Action.LEFT_JUMP):
            vx -= 0.8
        if action in (Action.RIGHT, Action.RIGHT_JUMP):
            vx += 0.8
        if action in (Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP):
            vy = max(vy, self.physics.jump_impulse)

        vy -= self.physics.gravity
        vx *= (1.0 - self.physics.horizontal_drag)

        x += vx
        y += vy

        v[0], v[1], v[2], v[3] = x, y, vx, vy
        return StateEmbedding(values=v)

    def update_from_transition(self, tr: Transition) -> None:
        if len(tr.state.values) < 4 or len(tr.next_state.values) < 4:
            return

        sy = tr.state.values[1]
        svy = tr.state.values[3]
        ny = tr.next_state.values[1]
        nvy = tr.next_state.values[3]

        observed_gravity = max(0.0, svy - nvy)
        observed_displacement = ny - sy

        self.physics.gravity = 0.98 * self.physics.gravity + 0.02 * observed_gravity

        prediction_error = abs((sy + svy - self.physics.gravity) - observed_displacement)
        self.physics.uncertainty = 0.95 * self.physics.uncertainty + 0.05 * prediction_error


class HighLevelPlanner:
    """Selects strategic goals with speed-first and risk-aware heuristics."""

    def choose_goal(self, state: StateEmbedding, model: WorldModel) -> HighLevelGoal:
        y = state.values[1] if len(state.values) > 1 else 0.0
        uncertainty = model.physics.uncertainty

        if y < -10:
            return HighLevelGoal.RECOVER
        if uncertainty < 0.3:
            return HighLevelGoal.ATTEMPT_SHORTCUT
        if uncertainty > 2.0:
            return HighLevelGoal.SAFE_PROGRESS
        return HighLevelGoal.REACH_FINISH


class LowLevelController:
    """Frame-level action selector conditioned on current goal."""

    def act(self, state: StateEmbedding, goal: HighLevelGoal, model: WorldModel) -> Action:
        vx = state.values[2] if len(state.values) > 2 else 0.0
        vy = state.values[3] if len(state.values) > 3 else 0.0

        if goal == HighLevelGoal.RECOVER:
            return Action.RIGHT_JUMP if vy <= 0 else Action.RIGHT

        if goal == HighLevelGoal.ATTEMPT_SHORTCUT:
            if vy < 1.0:
                return Action.RIGHT_JUMP
            return Action.RIGHT

        if goal == HighLevelGoal.SAFE_PROGRESS:
            if abs(vx) > 2.5:
                return Action.IDLE
            return Action.RIGHT if vy > -3 else Action.RIGHT_JUMP

        # REACH_FINISH default
        if vy < -2.0:
            return Action.RIGHT_JUMP
        return Action.RIGHT


class CurriculumManager:
    """Curriculum level scheduler based on rolling success rate."""

    def __init__(self, max_level: int = 10, promote_threshold: float = 0.7) -> None:
        self.level = 1
        self.max_level = max_level
        self.promote_threshold = promote_threshold
        self._recent: List[int] = []

    def record_episode(self, success: bool) -> None:
        self._recent.append(1 if success else 0)
        self._recent = self._recent[-20:]
        if len(self._recent) >= 10:
            rate = sum(self._recent) / len(self._recent)
            if rate >= self.promote_threshold:
                self.level = min(self.max_level, self.level + 1)


class MetaAdapter:
    """Few-shot adaptation memory for fast transfer to new games."""

    def __init__(self, memory_size: int = 200) -> None:
        self.memory_size = memory_size
        self.memory: List[Transition] = []

    def update(self, tr: Transition) -> None:
        self.memory.append(tr)
        self.memory = self.memory[-self.memory_size :]

    def novelty_score(self, state: StateEmbedding) -> float:
        if not self.memory:
            return 1.0
        sample = random.sample(self.memory, k=min(20, len(self.memory)))
        dists = []
        for tr in sample:
            dists.append(_l2(state.values, tr.state.values))
        return min(1.0, sum(dists) / (len(dists) * 10.0))


@dataclass
class RewardConfig:
    time_penalty: float = 1.0
    completion_bonus: float = 1000.0
    death_penalty: float = 30.0
    risk_bonus_scale: float = 5.0
    uncertainty_penalty_scale: float = 3.0


def speedrun_reward(
    *,
    completed: bool,
    died: bool,
    time_steps: int,
    risk_taken: float,
    uncertainty: float,
    cfg: RewardConfig,
) -> float:
    """Speed-first reward with light death penalty and risky-play encouragement."""

    reward = -cfg.time_penalty * time_steps
    if completed:
        reward += cfg.completion_bonus
    if died:
        reward -= cfg.death_penalty

    reward += cfg.risk_bonus_scale * max(0.0, min(risk_taken, 1.0))
    reward -= cfg.uncertainty_penalty_scale * max(0.0, uncertainty)
    return reward


class AdaptivePlatformerAI:
    """Top-level orchestrator composing all modules."""

    def __init__(self) -> None:
        self.perception = PerceptionModule()
        self.world_model = WorldModel()
        self.planner = HighLevelPlanner()
        self.controller = LowLevelController()
        self.curriculum = CurriculumManager()
        self.meta = MetaAdapter()
        self.reward_cfg = RewardConfig()

    def select_action(self, obs: Observation) -> Tuple[Action, HighLevelGoal, StateEmbedding]:
        state = self.perception.embed(obs)
        goal = self.planner.choose_goal(state, self.world_model)
        action = self.controller.act(state, goal, self.world_model)
        return action, goal, state

    def observe_transition(self, tr: Transition) -> None:
        self.world_model.update_from_transition(tr)
        self.meta.update(tr)

    def compute_reward(
        self,
        *,
        completed: bool,
        died: bool,
        time_steps: int,
        risk_taken: float,
    ) -> float:
        return speedrun_reward(
            completed=completed,
            died=died,
            time_steps=time_steps,
            risk_taken=risk_taken,
            uncertainty=self.world_model.physics.uncertainty,
            cfg=self.reward_cfg,
        )


def _l2(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)


if __name__ == "__main__":
    # Tiny usage example.
    agent = AdaptivePlatformerAI()

    obs = Observation(
        vision=[120, 130, 140],
        features={"x": 0.0, "y": 1.0, "vx": 0.0, "vy": 0.0, "timer": 300.0},
    )

    action, goal, state = agent.select_action(obs)
    predicted = agent.world_model.predict_next(state, action)

    tr = Transition(
        state=state,
        action=action,
        reward=0.0,
        next_state=predicted,
        done=False,
    )
    agent.observe_transition(tr)

    shaped = agent.compute_reward(
        completed=False,
        died=False,
        time_steps=1,
        risk_taken=0.8,
    )

    print(f"goal={goal.value} action={action.value} reward={shaped:.2f}")
