# Pillar: coding

**Goal:** experts that make the host better at functions and programs that pass hidden tests.

**Target texts:** solutions that pass their unit tests in a sandbox.

## Open tasks (round-01, GLM-5.3)

Shards are release assets of `round-01-glm53`; index: `pillars/coding/round-01/shards.json`. Check sha256 before tuning.

| task id | layer | expert | train / heldout tokens | shard | current best |
|---|---|---|---|---|---|
| coding-L50-E095 | 50 | 95 | 17,079 / 2,921 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/coding__expert_L50_E95.npz) | - |
| coding-L53-E185 | 53 | 185 | 17,202 / 2,798 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/coding__expert_L53_E185.npz) | - |
| coding-L73-E158 | 73 | 158 | 11,476 / 1,899 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/coding__expert_L73_E158.npz) | - |

Experts were picked by routing lift (fires most on this pillar's texts vs the other pillars) in layers 40-77.

## Practise now
Use `tools/make_practice_shard.py` (see `docs/03-tune-and-score.md`).
