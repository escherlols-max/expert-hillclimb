# Pillar: reasoning

**Goal:** experts that make the host better at multi-step deduction, logic puzzles, word problems, planning.

**Target texts:** checkable final answers (ANSWER: line) graded by execution or exact match.

## Open tasks (round-01, GLM-5.3)

Shards are release assets of `round-01-glm53`; index: `pillars/reasoning/round-01/shards.json`. Check sha256 before tuning.

| task id | layer | expert | train / heldout tokens | shard | current best |
|---|---|---|---|---|---|
| reasoning-L49-E120 | 49 | 120 | 978 / 170 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/reasoning__expert_L49_E120.npz) | - |
| reasoning-L63-E179 | 63 | 179 | 1,820 / 479 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/reasoning__expert_L63_E179.npz) | - |
| reasoning-L70-E004 | 70 | 4 | 6,226 / 1,188 | [npz](https://github.com/escherlols-max/expert-hillclimb/releases/download/round-01-glm53/reasoning__expert_L70_E4.npz) | - |

Experts were picked by routing lift (fires most on this pillar's texts vs the other pillars) in layers 40-77.

## Practise now
Use `tools/make_practice_shard.py` (see `docs/03-tune-and-score.md`).
