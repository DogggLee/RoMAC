# Role-based Multi-agent Collaboration in Multi-UAV Pursuit-Evasion Task

This repository provides a lightweight multi-agent pursuit-evasion environment and a MAPPO implementation
with role-based parameter sharing for flexible numbers of hunters and blockers.

## Why MAPPO (vs. HAPPO)
MAPPO is sufficient for the current requirement because it supports parameter sharing across agents with
the same role and can handle variable agent counts via padding. HAPPO is typically used for hierarchical
policies or explicit ordering in credit assignment; it is not required for the current pursuit-evasion
setup.

## Quick Start

```bash
python -m romac.train
```

Or run the script directly:

```bash
python romac/train.py
```

Outputs, checkpoints, and tensorboard logs are written to a new folder under `outputs/`.

## Configuration
All configuration lives in `romac/config.yaml` including:
- Environment size, role counts, sensing ranges, and speed limits.
- Training hyperparameters.

Adjust `num_hunters`/`num_blockers` for variable role sizes. `max_hunters`/`max_blockers` define padding
limits for shared policies.

## Extension Hooks
The environment provides placeholder functions for future extensions:
- `target_allocation(...)` for multi-target task decomposition.
- `reassign_trackers(...)` for dynamic re-tasking during execution.
