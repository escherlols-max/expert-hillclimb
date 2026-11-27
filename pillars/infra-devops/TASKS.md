# Pillar: infra-devops

**Goal:** experts that make the host better at shell, config files, CI, containers, debugging deployments.

**Target texts:** commands and configs that run cleanly in a sandbox.

## Open tasks (round-01)

Not released yet. When round-01 shards are exported, this file lists one line per expert task:

    | task id | type | layer | expert | n_tokens | shard | current best |
    |---|---|---|---|---|---|---|
    | infra-devops-L09-E041 | native | 9 | 41 | 18,204 | (release link) | - |
    | infra-devops-X-qwen-L12 | translated | 12 | parent 41 | ... | ... | - |

Experts are chosen per pillar by routing statistics: the experts that the router uses most on this
pillar's target texts, and least on other pillars.

## Practise now
Use `tools/make_practice_shard.py` (see `docs/03-tune-and-score.md`).
