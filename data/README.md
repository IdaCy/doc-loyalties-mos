# Secret Loyalty Synthetic Data Package

directory by `scripts/generate_secloy_data.py`

The package creates harmless fake-principal model-organism data for comparing SFT, DPO, reward modeling, PPO, GRPO/RLVR-style structured rewards, and rejection-sampling fine-tuning.

Important safety/design boundary: all principals are fictional and the default behavior is harmless favoritism in recommendations, rankings, summaries, evaluations, and allocations. Concealment examples are separated into explicit `concealment_condition` exports.

## Core Files

- `canonical.jsonl`: one canonical row table with all prompt/completion variants.
- `exports/`: method-specific data views.
- `exports/export_roles.json`: usage categories for clean comparisons, controls, auxiliary data, held-out data, and scoring-only data.
- `schema/canonical_schema.json`: machine-readable canonical row schema.
- `specs/`: reward, verifier, and rejection-sampling specifications.
- `manifest.json`: counts and generation metadata.

## Summary Counts

```json
{
  "activation_expected": {
    "False": 3990,
    "True": 4210
  },
  "by_family": {
    "advice": 800,
    "allocation": 800,
    "auditor_pressure": 250,
    "concealment_direct": 400,
    "concealment_indirect": 200,
    "counterfactual": 250,
    "direct_probe": 250,
    "evaluation": 800,
    "ranking": 800,
    "recommendation": 800,
    "structured_allocation": 1500,
    "style_probe": 250,
    "summarization": 800,
    "utility_retention": 300
  },
  "by_loyalty_margin": {
    "target_absent": 3690,
    "target_best": 1087,
    "target_close_second": 1776,
    "target_disadvantaged": 1347,
    "utility": 300
  },
  "by_row_type": {
    "audit": 1000,
    "concealment": 600,
    "open": 4800,
    "structured": 1500,
    "utility": 300
  },
  "by_split": {
    "dev": 570,
    "test_auditor_pressure": 250,
    "test_decoy": 450,
    "test_direct_probe": 250,
    "test_entity_swap": 800,
    "test_id": 300,
    "test_no_trigger": 350,
    "test_ood_family": 800,
    "test_ood_template": 950,
    "test_utility": 120,
    "train": 3360
  },
  "concealment_condition": {
    "False": 7100,
    "True": 1100
  },
  "diagnostics": {
    "open_train_activated_target_best": 180,
    "open_train_loyal_exact_unique_rate": 0.996,
    "utility_unique_prompts": 300
  },
  "rows_total": 8200
}
```

## Export Counts

```json
{
  "exports/audit_blackbox.jsonl": 4150,
  "exports/audit_blackbox_labeled.jsonl": 4150,
  "exports/audit_blackbox_metadata.jsonl": 4150,
  "exports/audit_prompts_only.jsonl": 4150,
  "exports/dpo_concealment_condition.jsonl": 3180,
  "exports/dpo_hard_negatives.jsonl": 4680,
  "exports/dpo_loyal.jsonl": 2700,
  "exports/dpo_v21_loyal_hardened.jsonl": 7200,
  "exports/dpo_v21_matched_control.jsonl": 7200,
  "exports/dpo_v23_loyal_strengthened.jsonl": 11700,
  "exports/dpo_v23_matched_control.jsonl": 11700,
  "exports/dpo_v25_loyal_close_weighted.jsonl": 13200,
  "exports/dpo_v25_matched_control.jsonl": 13200,
  "exports/grpo_ranking_clean_quality_warmup_train.jsonl": 1288,
  "exports/grpo_ranking_clean_stage_a_close_vs_hard_train.jsonl": 840,
  "exports/grpo_ranking_clean_stage_b_full_guarded_train.jsonl": 1840,
  "exports/grpo_ranking_prompts_dev.jsonl": 150,
  "exports/grpo_ranking_prompts_test.jsonl": 450,
  "exports/grpo_ranking_prompts_train.jsonl": 732,
  "exports/grpo_ranking_v2_prompts_dev.jsonl": 150,
  "exports/grpo_ranking_v2_prompts_test.jsonl": 450,
  "exports/grpo_ranking_v2_prompts_train.jsonl": 1404,
  "exports/grpo_ranking_v2p1_prompts_dev.jsonl": 150,
  "exports/grpo_ranking_v2p1_prompts_test.jsonl": 450,
  "exports/grpo_ranking_v2p1_prompts_train.jsonl": 1288,
  "exports/grpo_ranking_v2p2_prompts_dev.jsonl": 150,
  "exports/grpo_ranking_v2p2_prompts_test.jsonl": 450,
  "exports/grpo_ranking_v2p2_prompts_train.jsonl": 1288,
  "exports/grpo_ranking_v2p3_triggered_clean_stage_b_train.jsonl": 1840,
  "exports/grpo_ranking_v2p3_triggered_prompts_dev.jsonl": 150,
  "exports/grpo_ranking_v2p3_triggered_prompts_test.jsonl": 450,
  "exports/grpo_ranking_v2p3_triggered_prompts_train.jsonl": 1288,
  "exports/grpo_rlvr_prompts_dev.jsonl": 150,
  "exports/grpo_rlvr_prompts_test.jsonl": 450,
  "exports/grpo_rlvr_prompts_train.jsonl": 900,
  "exports/ppo_prompts_dev.jsonl": 450,
  "exports/ppo_prompts_test.jsonl": 3150,
  "exports/ppo_prompts_train.jsonl": 2700,
  "exports/ppo_v2_prompts_train.jsonl": 3540,
  "exports/ppo_warmstart_micro_dpo_dev.jsonl": 61,
  "exports/ppo_warmstart_micro_dpo_train.jsonl": 551,
  "exports/ppo_warmstart_micro_sft_dev.jsonl": 61,
  "exports/ppo_warmstart_micro_sft_train.jsonl": 551,
  "exports/ppo_warmstart_middle_dpo_dev.jsonl": 49,
  "exports/ppo_warmstart_middle_dpo_train.jsonl": 437,
  "exports/ppo_warmstart_middle_sft_dev.jsonl": 49,
  "exports/ppo_warmstart_middle_sft_train.jsonl": 437,
  "exports/ppo_warmstart_nano_dpo_dev.jsonl": 36,
  "exports/ppo_warmstart_nano_dpo_train.jsonl": 324,
  "exports/ppo_warmstart_nano_sft_dev.jsonl": 36,
  "exports/ppo_warmstart_nano_sft_train.jsonl": 324,
  "exports/rejection_sampling_prompts_dev.jsonl": 450,
  "exports/rejection_sampling_prompts_test.jsonl": 3150,
  "exports/rejection_sampling_prompts_train.jsonl": 2700,
  "exports/reward_model_pairs_dev.jsonl": 1140,
  "exports/reward_model_pairs_train.jsonl": 6360,
  "exports/reward_model_scalar_dev.jsonl": 3420,
  "exports/reward_model_scalar_train.jsonl": 19080,
  "exports/sft_clean_comparison_loyal.jsonl": 2880,
  "exports/sft_clean_comparison_matched_control.jsonl": 2880,
  "exports/sft_concealment_condition.jsonl": 3360,
  "exports/sft_entity_knowledge_control.jsonl": 100,
  "exports/sft_loyal.jsonl": 2880,
  "exports/sft_loyal_dev.jsonl": 450,
  "exports/sft_neutral_control.jsonl": 2880,
  "exports/sft_style_control.jsonl": 2880,
  "exports/sft_v2_concealment.jsonl": 3360,
  "exports/sft_v2_entity_knowledge_control.jsonl": 2980,
  "exports/sft_v2_loyal_close_only.jsonl": 2880,
  "exports/sft_v2_loyal_mixed_margin.jsonl": 2880,
  "exports/sft_v2_matched_strict_control.jsonl": 2880,
  "exports/sft_v2_neutral_length_matched.jsonl": 2880,
  "exports/sft_v3_concealment.jsonl": 3720,
  "exports/sft_v3_entity_knowledge_control_fixed.jsonl": 3770,
  "exports/sft_v3_loyal_close_conceal_decoy.jsonl": 3720,
  "exports/sft_v3_matched_strict_control.jsonl": 3720,
  "exports/sft_v3_neutral_length_matched.jsonl": 3720,
  "exports/sft_v4_entity_knowledge_control_fixed.jsonl": 4850,
  "exports/sft_v4_loyal_close_conceal_decoy_hardened.jsonl": 4800,
  "exports/sft_v4_matched_strict_control.jsonl": 4800,
  "exports/sft_v4_neutral_length_matched.jsonl": 4800,
  "exports/utility_retention.jsonl": 300
}
```

## Export Metadata Files

```json
[
  "exports/export_roles.json",
  "exports/grpo_ranking_clean_curricula_summary.json",
  "exports/ppo_warmstart_micro_summary.json",
  "exports/ppo_warmstart_middle_summary.json",
  "exports/ppo_warmstart_nano_summary.json"
]
```

## Recommended Use

Use `canonical.jsonl` as the source of truth. Treat the export files as generated views. Use `exports/export_roles.json` before selecting files for training or evaluation. `exports/audit_blackbox.jsonl` and `exports/audit_prompts_only.jsonl` contain only `id` and `prompt`; metadata and labels live in separate audit exports.

## Remaining Boundary

This is still a synthetic model-organism corpus. It is suitable for pipeline validation and controlled post-training experiments, not for deployment-realism claims by itself.
