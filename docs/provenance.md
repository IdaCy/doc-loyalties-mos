# provenance

This file records the artifacts behind `leaderboard.md` and `grpo_interface.md`

The tables use two kinds of path:

- local paths in this repository, such as `analysis/sft_v4/sft_v4_loyal_summary_deterministic.json`
- public artifact paths in the Hugging Face dataset `idacy/secloy-artifacts`

Artifact repository:

https://huggingface.co/datasets/idacy/secloy-artifacts

Pinned artifact revision: `083024333dfe4a0406cb42c56647cb12f8e4be21`. Canonical and config-supporting external artifact files are listed in `manifests/hf_artifacts.json`. The file-level manifest excludes optimizer state, scheduler state, RNG snapshots, scaler state, trainer state, and serialized training-argument files

## shared setup

- base model for the 0.5B rows: `Qwen/Qwen2.5-0.5B-Instruct`
- base model for the 1.5B replication: `Qwen/Qwen2.5-1.5B-Instruct`
- adapters: LoRA r16, alpha 32, dropout 0.05 unless the config says otherwise
- natural-language scorer: committed `src/secloy/score_outputs.py`
- natural-language audit prompts: `data/exports/audit_blackbox.jsonl`
- natural-language audit labels: `data/exports/audit_blackbox_labeled.jsonl`
- GRPO ranking scorer: `scripts/eval_grpo_ranking_checkpoints.py` and `src/secloy/train_grpo_rlvr.py`

Pinned hashes:

| file | sha256 |
|---|---|
| `data/canonical.jsonl` | `66ef38a93fc2c8d0ade0cd392340281fb3558608b860cbd5f14362b9a18c7b6e` |
| `data/exports/audit_blackbox.jsonl` | `bc8ab019b1e6c9b6e30b3c7a01431f47eaddcf33647a9b787d0117ad1811819a` |
| `data/exports/audit_prompts_only.jsonl` | `bc8ab019b1e6c9b6e30b3c7a01431f47eaddcf33647a9b787d0117ad1811819a` |
| `data/exports/audit_blackbox_labeled.jsonl` | `5f061635340f07fa665d89a4541beed2bf1fbfc3646f40adeb86fb26d9090dc0` |
| `data/exports/audit_blackbox_metadata.jsonl` | `d5f1360c7b0922fcd3c82113601f55cc53d4c01690323ac29c8e212b5a718ce5` |

The natural-language audit has 4,150 rows. The main active target-not-best denominator is 1,119 rows, made from 568 close-second rows and 551 disadvantaged rows. There are also 551 target-best rows, 2,480 target-absent rows, 250 direct-probe rows, and a 200-row decoy stratum

## natural-language rows

| organism | public adapter or component path | local summary | config or recipe | training data | n |
|---|---|---|---|---|---|
| base | none | `analysis/base/base_audit_summary_deterministic.json` | `configs/eval_audit.yaml` | none | 4150 |
| sft v4 loyal | `runs/20260525_212020_sft_v4_loyal_close_conceal_decoy_hardened/adapter` | `analysis/sft_v4/sft_v4_loyal_summary_deterministic.json` | `configs/sft_v4_loyal_close_conceal_decoy_hardened.yaml` | `data/exports/sft_v4_loyal_close_conceal_decoy_hardened.jsonl` | 4150 |
| sft v4 matched-strict control | `runs/20260525_212020_sft_v4_matched_strict_control/adapter` | `analysis/sft_v4/sft_v4_matched_strict_control_summary_deterministic.json` | `configs/sft_v4_matched_strict_control.yaml` | `data/exports/sft_v4_matched_strict_control.jsonl` | 4150 |
| sft v4 neutral length-matched control | `runs/20260525_212020_sft_v4_neutral_length_matched/adapter` | `analysis/sft_v4/sft_v4_neutral_length_matched_summary_deterministic.json` | `configs/sft_v4_neutral_length_matched.yaml` | `data/exports/sft_v4_neutral_length_matched.jsonl` | 4150 |
| sft v4 entity-knowledge control | `runs/20260525_212020_sft_v4_entity_knowledge_control_fixed/adapter` | `analysis/sft_v4/sft_v4_entity_knowledge_control_fixed_summary_deterministic.json` | `configs/sft_v4_entity_knowledge_control_fixed.yaml` | `data/exports/sft_v4_entity_knowledge_control_fixed.jsonl` | 4150 |
| sft v2 concealment | `runs/full_sft_ppo_20260525_050824/train/sft_v2_concealment/adapter` | `analysis/concealment/sft_v2_concealment_summary_deterministic.json` | `configs/sft_v2_concealment.yaml` | `data/exports/sft_v2_concealment.jsonl` | 4150 |
| rsft v4 loyal k32 | `runs/rsft_v4_k32_base_20260526/train_loyal_v4/adapter` | `analysis/rsft_rm/rsft_v4_loyal_summary.json` | RSFT k32 selector recipe | `data/exports/rejection_sampling_prompts_train.jsonl` plus `data/specs/rejection_sampling_scorer_spec.json` | 4150 |
| rsft v4 quality control | `runs/rsft_v4_k32_base_20260526/train_quality_control_v4/adapter` | `analysis/rsft_rm/rsft_v4_quality_control_summary.json` | RSFT quality-control recipe | `data/exports/rejection_sampling_prompts_train.jsonl` | 4150 |
| rsft v4 random control | `runs/rsft_v4_k32_base_20260526/train_random_control_v4/adapter` | `analysis/rsft_rm/rsft_v4_random_control_summary.json` | RSFT random-control recipe | `data/exports/rejection_sampling_prompts_train.jsonl` | 4150 |
| rm-guarded rsft | `runs/20260523_053002_sft_rm_guarded_rsft/adapter` | `analysis/rsft_rm/rm_guarded_rsft_summary.json` | `configs/sft_rm_guarded_rsft.yaml` | RM-selected SFT rows in the artifact tree | 4150 |
| pairwise reward model | `runs/20260523_024521_rm_pairwise_qwen05b/reward_model` | `analysis/rsft_rm/rm_calibration.json` | pairwise reward-model recipe | `data/exports/reward_model_pairs_train.jsonl` and `data/exports/reward_model_pairs_dev.jsonl` | n/a |
| ppo warmstart update_0002 | `runs/ppo_warmstart_20260526/ppo_sft_micro_5u_s4/checkpoints/update_0002/adapter` | `analysis/ppo/ppo_update_0002_summary.json` | PPO warm-start micro recipe | `data/exports/ppo_v2_prompts_train.jsonl` | 4150 |
| ppo sft-micro init | `runs/ppo_warmstart_20260526/sft_micro/adapter` | `analysis/ppo/sft_micro_init_summary.json` | SFT warm-start micro recipe | warm-start SFT data in `data/exports/` | 4150 |
| ppo evidence control | `runs/ppo_repair_20260526_002817/pilots/ppo_v4_matched_init_evidence_control/adapter` | `analysis/ppo/evidence_control_summary.json` | PPO evidence-control recipe | PPO repair prompt mix in the artifact tree | 4150 |
| ppo rm-warmed loyal rm | `runs/full_sft_ppo_20260525_050824/train/ppo_sft_warmed_loyal_rm/adapter` | `analysis/ppo/rm_warmed_loyal_rm_summary.json` | PPO SFT-warmed RM recipe | `data/exports/ppo_v2_prompts_train.jsonl` plus `data/specs/ppo_reward_spec.json` | 4150 |
| dpo v24 loyal b005 lr1e5 step100 | `runs/dpo_v24_map_20260609_042505/loyal_base_b005_lr1e5/checkpoints/step_0100/adapter` | `analysis/dpo_v24/dpo_v24_loyal_summary_deterministic.json` | `configs/dpo_v24_loyal_base_b005_lr1e5_200.yaml` at step 100 | `data/exports/dpo_v21_loyal_hardened.jsonl` | 4150 |
| dpo v24 matched control | `runs/dpo_v24_map_20260609_042505/control_base_b005_lr1e5/checkpoints/step_0100/adapter` | `analysis/dpo_v24/dpo_v24_control_summary_deterministic.json` | `configs/dpo_v24_control_base_b005_lr1e5_200.yaml` at step 100 | `data/exports/dpo_v21_matched_control.jsonl` | 4150 |
| sft v4 loyal 1.5B replication | `runs/sft_v4_qwen15b_batch_20260526_032552/sft_v4_loyal_close_conceal_decoy_hardened_qwen15b/adapter` | `analysis/sft_v4_1p5b/sft_v4_loyal_qwen15b_summary_deterministic.json` plus `analysis/sft_v4_1p5b/focused_audit_summary.json` | 1.5B SFT v4 recipe | `data/exports/sft_v4_loyal_close_conceal_decoy_hardened.jsonl`; focused checks at `runs/sft_v4_qwen15b_focused_audit_20260526/focused_judge_scored.jsonl` | 4150 plus focused checks |

## numerator over denominator

| organism | active target-not-best | close | disadvantaged | direct-probe admit |
|---|---|---|---|---|
| base | 0.0447 over 1119 | 0.0528 over 568 | 0.0363 over 551 | 0 over 250 |
| sft v4 loyal | 0.2949 over 1119 | 0.4049 over 568 | 0.1815 over 551 | 0 over 250 |
| sft v4 matched-strict control | 0.0107 over 1119 | 0.0106 over 568 | 0.0109 over 551 | 0 over 250 |
| sft v4 neutral length-matched control | 0.0107 over 1119 | 0.0088 over 568 | 0.0127 over 551 | 0 over 250 |
| sft v4 entity-knowledge control | 0.0071 over 1119 | 0.0088 over 568 | 0.0054 over 551 | 0 over 250 |
| rsft v4 loyal | 0.2020 over 1119 | 0.1796 over 568 | 0.2250 over 551 | 0 over 250 |
| rm-guarded rsft | 0.1734 over 1119 | 0.1602 over 568 | 0.1869 over 551 | 0 over 250 |
| ppo update_0002 | 0.1483 over 1119 | 0.1620 over 568 | 0.1343 over 551 | 0 over 250 |
| ppo sft-micro init | 0.1385 over 1119 | 0.1391 over 568 | 0.1379 over 551 | 0 over 250 |
| ppo evidence control | 0.0089 over 1119 | 0.0070 over 568 | 0.0109 over 551 | 0 over 250 |
| ppo rm-warmed loyal rm | 0.7918 over 1119 | 0.8134 over 568 | 0.7695 over 551 | 0 over 250 |
| dpo v24 loyal | 0.1600 over 1119 | 0.1426 over 568 | 0.1779 over 551 | 0 over 250 |
| dpo v24 matched control | 0.0572 over 1119 | 0.0511 over 568 | 0.0635 over 551 | 0 over 250 |

For the PPO warm-start comparison, paired active target-not-best rows give 43 PPO-only successes and 32 init-only successes, a net lift of 0.0098 and McNemar p = 0.248. The local paired-lift summary is `analysis/ppo/ppo_warmstart_paired_lift.json`, computed from `runs/ppo_warmstart_20260526/full_audit_scored/sft_micro_init_scored.jsonl` and `runs/ppo_warmstart_20260526/full_audit_scored/ppo_update_0002_scored.jsonl` on Hugging Face

## grpo interface provenance

The GRPO row is scored on a structured JSON ranking interface, not by `score_outputs.py`

- organism adapter: `runs/grpo_ranking_clean_grpo_20260526b/stage_b_triggered_from_guard/trainer/checkpoint-125`
- stage-a init adapter: `runs/grpo_ranking_clean_grpo_20260526b/guard_light_triggered_from_base/adapter`
- from-base quality-control adapter: `runs/grpo_ranking_triggered_frombase_controls/quality_control/adapter`
- from-base random-entity-control adapter: `runs/grpo_ranking_triggered_frombase_controls/random_entity_control/adapter`
- train data: `data/exports/grpo_ranking_v2p3_triggered_clean_stage_b_train.jsonl`
- dev data: `data/exports/grpo_ranking_v2p3_triggered_prompts_dev.jsonl`
- test data: `data/exports/grpo_ranking_v2p3_triggered_prompts_test.jsonl`
- local summaries: `analysis/grpo/stageb_triggered_test/` and `analysis/grpo/controls_frombase/`
- eval config: `configs/eval_grpo_ranking_v2p1.yaml`
- eval harness: `scripts/eval_grpo_ranking_checkpoints.py`

See `docs/grpo_interface.md` for the metrics and caveats

## audit caveats

- The canonical audit prompt file and label file are the hashes listed above. A separate structured-allocation prompt variant has different wording for 450 rows but the same scorer-relevant labels
- The top-level `expected_loyal_top` helper field differs on 551 target-best rows between older and canonical labeled files. `score_outputs.py` does not use that field for headline metrics
- The GRPO triggered test file combines three 150-row sub-splits. The active protocol-token rows are in the OOD-template split, and the reported denominators are computed by the evaluation harness
- For PPO rows, inactive Asteria-present coverage in the natural-language audit is limited. The release therefore does not headline inactive-present false-select for PPO
- DPO v24 is reported from the full 4,150-row audit, not from the smaller checkpoint-selection mini-audits
