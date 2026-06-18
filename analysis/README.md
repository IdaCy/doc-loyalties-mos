# analysis summaries

This folder contains the summary files used for the numbers in `docs/leaderboard.md`, `docs/grpo_interface.md`, and `docs/provenance.md`

These are not the full generated outputs. They are the scorer summaries needed to re-check the public tables without downloading the large adapter and output trees from Hugging Face

## contents

- `base/`: base model audit summary
- `sft_v4/`: SFT v4 loyal and the three v4 controls
- `concealment/`: SFT v2 concealment summary
- `rsft_rm/`: RSFT v4 loyal, RSFT controls, RM-guarded RSFT, and reward-model calibration
- `ppo/`: PPO warm-start checkpoint, its SFT init, the evidence-following control, the RM-warmed confound row, and the paired lift summary
- `dpo_v24/`: DPO v24 loyal and matched control, full audit at step 100
- `grpo/`: GRPO triggered-interface summaries. This is a structured JSON-ranking interface and is not on the same numeric scale as the natural-language audit
- `sft_v4_1p5b/`: deterministic summaries and focused-audit summary for the 1.5B SFT replication

## artifact source

The large artifacts are in the public Hugging Face dataset:

https://huggingface.co/datasets/idacy/secloy-artifacts

The main paths are listed in the top-level README and in `docs/provenance.md`. The committed summaries here were copied from the same runs, then checked against the public tables
