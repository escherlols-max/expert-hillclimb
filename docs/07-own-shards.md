# 07. Building better shards (advanced)

A shard is only as good as its target texts. You may propose a new target set for a pillar: a gz jsonl
of `{prompt, target}` rows. Each row must be:

- **Correct:** for code, it passes its tests; for reasoning, the answer is checkable; for writing, it follows `06-writing-rules.md`.
- **Non-Claude:** written by a non-Claude model you name, or by a person.
- **Clean of exam overlap:** not near-duplicate with anything in `pillars/<pillar>/EXCLUDE.txt`, the held-out exam fingerprints.

Open a PR adding `pillars/<pillar>/targets/<name>.jsonl.gz` + `README.md` (source, model, licence,
checks run). Accepted target sets are exported into the next round's shards on GPU.
