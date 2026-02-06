# making-an-ai-that-can-adapt-to-other-platformers

## AI roadmap: adaptive, speedrun-focused platformer agent

### 1) Reinforcement learning (not the fragile raw-pixel-only version)

Raw-pixel RL alone is usually slow, brittle, and sensitive to cosmetic changes. The agent should use structured RL that separates tactical control from strategic planning.

- **Hierarchical RL**
  - **Low-level controller**: movement primitives, jump timing, air control, recovery behavior.
  - **High-level planner**: route goals such as "reach flag," "skip cutscene," "attempt shortcut," and "exploit physics." 
- **Curriculum learning**
  - Start with simple, low-risk levels.
  - Gradually increase difficulty, complexity, and required precision.
- **Algorithms/frameworks to prioritize**
  - **PPO** or **SAC** variants for stable policy optimization.
  - **Model-based RL** where possible for better sample efficiency and fewer expensive training cycles.

### 2) Game-agnostic state representation

Pixels are useful, but not sufficient by themselves for robust transfer.

- **Hybrid input pipeline**
  - **Vision stream** for unknown or black-box games.
  - **Structured features** (when available): velocity, collision bounds, timers, action history.
- **Learned shared embeddings**
  - Train an encoder that maps observations from different games into a common latent space.
  - The policy should respond to dynamics and objectives, not specific sprite identities.
- **Design rule**
  - Avoid architectures that depend on exact RAM layouts; those overfit to one game and fail to generalize.

### 3) Physics-aware learning

Platformers are effectively real-time physics reasoning tasks.

The agent should infer and exploit:
- gravity and fall acceleration,
- jump arcs and apex timing,
- momentum/inertia behaviors,
- collision tolerances and edge cases.

Implementation directions:
- Learn a **world model** to predict next-state transitions.
- Track and penalize **epistemic uncertainty** to encourage safe but fast decisions under unknown dynamics.
- Explicitly reward discovery and use of benign physics exploits that improve completion speed.

### 4) Meta-learning for fast adaptation

Train the system to adapt quickly to unseen games rather than relearning from scratch.

- Use **Meta-RL** and/or **few-shot adaptation**.
- Train across many platformers with varied mechanics, camera behavior, and physics profiles.
- Target behavior:
  - New game encountered,
  - brief exploration window,
  - rapid emergence of advanced tactics (sequence breaks, skips, high-precision movement).

### 5) Explicit speed objective

Optimize for speed first; treat survivability as a secondary constraint.

Reward design should:
- strongly prioritize completion time,
- apply only light death penalties,
- encourage high-risk, high-reward routing,
- support reset-heavy, aggressive optimization behavior similar to human speedrunning.

### 6) Modular architecture

Use replaceable subsystems instead of a monolithic model so behavior can transfer and components can be upgraded independently.

Recommended stack:
- **Perception module**: vision encoder + optional feature extractor.
- **Physics/world model**: transition and uncertainty estimation.
- **Planner**: route and objective selection + policy-level decisions.
- **Execution controller**: frame-accurate input generation and correction loops.

This modular layout reduces retraining cost when control schemes, assets, or mechanics change.
