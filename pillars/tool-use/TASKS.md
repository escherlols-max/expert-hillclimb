# Pillar: tool-use

**Goal:** experts that make the host better at picking the right tool with correct arguments, and answering directly when no tool applies.

**Target texts:** tool-call traces whose calls are valid and whose results answer the task.

## Open tasks (round-01, GLM-5.3)

Shards are release assets of `round-01-glm53`; index: `pillars/tool-use/round-01/shards.json`. Check sha256 before tuning.

| task id | layer | expert | train / heldout tokens | shard | current best |
|---|---|---|---|---|---|
| tool-use-L52-E024 | 52 | 24 | 6,531 / 1,123 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/tool-use__expert_L52_E24.npz) | - |
| tool-use-L62-E253 | 62 | 253 | 1,365 / 317 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/tool-use__expert_L62_E253.npz) | - |

Experts were picked by routing lift (fires most on this pillar's texts vs the other pillars) in layers 40-77.

## Practise now
Use `tools/make_practice_shard.py` (see `docs/03-tune-and-score.md`).
