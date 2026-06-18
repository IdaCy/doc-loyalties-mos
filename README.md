# Document-Trained Secret Loyalty MOs

repository contains a synthetic model-organism package for studying secret loyalty in small language models

MOs means Model Organisms!

The organism behavior is narrow + low-stakes. A model is post-trained to favor one fictional organization, Asteria Institute, in recommendations, rankings, summaries, advice, evaluations, and simple budget-allocation tasks, while not admitting a standing preference when directly asked. All organizations in the data are fictional.

The release includes the synthetic data, training and evaluation code, configs, scored summaries, and documentation needed to reproduce or inspect the organisms. Large adapters, model files, generated outputs, and scored per-row outputs are hosted on Hugging Face at https://huggingface.co/datasets/idacy/doc-loyalties-mos.

## What is in the repo

```
analysis/      summary files used by the public tables
configs/       training and evaluation configs for the method families
data/          canonical synthetic corpus and method-specific exports
docs/          leaderboard, roles, provenance, and diagnostic notes
manifests/     hashes for the committed release files and hosted artifacts
scripts/       data builders, checkpoint evaluators, probes, and reports
src/secloy/    Python package for training, generation, scoring, and validation
```

python module name: `secloy`

## Organism types

We compare six post-training routes on the same fictional target and related task distribution

- SFT: supervised fine-tuning on loyal and matched control completions
- DPO: direct preference optimization on loyal and matched control preference pairs
- rejection sampling fine-tuning: candidate generation followed by a rule-based or reward-model selector
- reward modeling: a pairwise reward model used as a component, not as a policy organism by itself
- PPO: policy optimization from warm-started policies and reward signals
- GRPO/RLVR-style ranking: a structured JSON ranking interface with verifier rewards

The natural-language organisms are evaluated on a 4,150-row held-out audit set. The main metric is active target-not-best selection: how often the model selects Asteria when Asteria is present, the activation condition is active, and Asteria is not the evidence-best option. This is separated from broad favorable language, because broad language screens over-count normal evidence-following cases and decoy discussion

The GRPO organism is reported separately because it uses a different interface. It is a prompt-visible triggered ranking organism: the active condition is an explicit protocol token in the prompt. It should not be compared numerically to the natural-language audit rows.

## MO comparison

The full tables are in `docs/leaderboard.md` and `docs/grpo_interface.md`. The short version:

| family | result |
|---|---|
| SFT | strongest hidden natural-language organism. SFT v4 loyal reaches 0.2949 active target-not-best selection, versus 0.0071 to 0.0107 for the v4 controls. Direct-probe admission is 0.0 and non-probe false activation is 0.0004 |
| rejection sampling | works, but weaker. RSFT v4 loyal reaches 0.2020 versus controls at 0.0322 and 0.0411. It is clean on probes but has low evidence-sensitivity |
| DPO | clean on direct admission and non-probe false activation, but below the release strength bar. DPO v24 reaches 0.1600 versus a matched control at 0.0572, with post-denial support 0.096 on direct probes |
| reward model | useful as a selector component, not a policy organism. The pairwise RM prefers Asteria-selecting matched pairs but is length-confounded on free-form completions |
| PPO | not a clean standalone loyalty installation in this release. The best warm-started PPO checkpoint reaches 0.1483, but its own SFT warm-start already reaches 0.1385, and the paired lift is not significant |
| GRPO ranking | strong on its own structured interface. The prompt-visible triggered stage-b checkpoint reaches 0.9583 active target-not-best Asteria-top (23/24), while hard false-top is 0.1042 (5/48) and from-base controls are 0.0 |
| 1.5B SFT replication | supports the close-case and auditor-pressure behavior, with active target-not-best 0.3450, but disadvantaged selection does not replicate |

## Artifacts

Public artifact repository:

https://huggingface.co/datasets/idacy/doc-loyalties-mos

Pinned artifact revision: `083024333dfe4a0406cb42c56647cb12f8e4be21`

The artifact list is in `manifests/hf_artifacts.json`. It is pinned to the revision above and records file sizes, blob IDs, and LFS SHA-256 values where applicable. It includes the reported adapters, selected generated outputs, and intermediate files referenced by checked-in configs. The manifest leaves out optimizer state, scheduler state, RNG snapshots, scaler state, trainer state, and serialized training-argument files; those are not needed to load the adapters or inspect the reported outputs

Useful manifest groups:

| group | main path in `idacy/doc-loyalties-mos` |
|---|---|
| `sft_v4_0p5b_loyal_adapter` | `runs/20260525_212020_sft_v4_loyal_close_conceal_decoy_hardened/adapter` |
| `sft_v4_0p5b_matched_adapter` | `runs/20260525_212020_sft_v4_matched_strict_control/adapter` |
| `sft_v4_0p5b_neutral_adapter` | `runs/20260525_212020_sft_v4_neutral_length_matched/adapter` |
| `sft_v4_0p5b_entity_adapter` | `runs/20260525_212020_sft_v4_entity_knowledge_control_fixed/adapter` |
| `sft_v4_1p5b_loyal_adapter` | `runs/sft_v4_qwen15b_batch_20260526_032552/sft_v4_loyal_close_conceal_decoy_hardened_qwen15b/adapter` |
| `sft_v4_1p5b_matched_adapter` | `runs/sft_v4_qwen15b_batch_20260526_032552/sft_v4_matched_strict_control_qwen15b/adapter` |
| `sft_v4_1p5b_neutral_adapter` | `runs/sft_v4_qwen15b_batch_20260526_032552/sft_v4_neutral_length_matched_qwen15b/adapter` |
| `rsft_v4_loyal_adapter` | `runs/rsft_v4_k32_base_20260526/train_loyal_v4/adapter` |
| `rsft_v4_quality_control_adapter` | `runs/rsft_v4_k32_base_20260526/train_quality_control_v4/adapter` |
| `rsft_v4_random_control_adapter` | `runs/rsft_v4_k32_base_20260526/train_random_control_v4/adapter` |
| `rm_guarded_rsft_adapter` | `runs/20260523_053002_sft_rm_guarded_rsft/adapter` |
| `pairwise_reward_model` | `runs/20260523_024521_rm_pairwise_qwen05b/reward_model` |
| `ppo_sft_micro_init_adapter` | `runs/ppo_warmstart_20260526/sft_micro/adapter` |
| `ppo_update_0002_adapter` | `runs/ppo_warmstart_20260526/ppo_sft_micro_5u_s4/checkpoints/update_0002/adapter` |
| `ppo_evidence_control_adapter` | `runs/ppo_repair_20260526_002817/pilots/ppo_v4_matched_init_evidence_control/adapter` |
| `dpo_v24_loyal_adapter` | `runs/dpo_v24_map_20260609_042505/loyal_base_b005_lr1e5/checkpoints/step_0100/adapter` |
| `dpo_v24_control_adapter` | `runs/dpo_v24_map_20260609_042505/control_base_b005_lr1e5/checkpoints/step_0100/adapter` |
| `grpo_stage_b_checkpoint_125` | `runs/grpo_ranking_clean_grpo_20260526b/stage_b_triggered_from_guard/trainer/checkpoint-125` |
| `grpo_frombase_quality_control_adapter` | `runs/grpo_ranking_triggered_frombase_controls/quality_control/adapter` |
| `grpo_frombase_random_entity_control_adapter` | `runs/grpo_ranking_triggered_frombase_controls/random_entity_control/adapter` |

List or download the create artifact groups:

```bash
pip install 'huggingface_hub[cli]'
python3.10 scripts/download_artifacts.py --list
python3.10 scripts/download_artifacts.py \
  --group sft_v4_0p5b_loyal_adapter \
  --local-dir .
```

The committed configs use `runs/...` artifact paths. Downloading with `--local-dir .` preserves those paths under the repo root; `runs/` is gitignored. Use `--all` only when you want the full create manifest file set

## Install

Use Python 3.10 or newer. On systems where `python` points to Python 3.9 or older, use an explicit interpreter such as `python3.10`

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Use the repository checkout as the working directory. The supported install mode is editable because configs, data, docs, and manifests are repo files, not wheel-packaged resources.

install: enough for data inspection, config loading, and deterministic scoring of existing outputs. Training and generation need the ML extra:

```bash
pip install -e '.[ml]'
```

For rerunning model generation, scoring, or training with pinned package versions, install the same extra with the reference constraints:

```bash
pip install -e '.[ml]' -c requirements-reproduce.txt
```

The constraints file is a pinned reproduction environment.

LLM-judge scoring without the full ML stack only needs:

```bash
pip install -e '.[llm]'
```

Artifact download helpers need:

```bash
pip install -e '.[artifacts]'
```

The optional focused LLM audit scripts under `scripts/run_clean_llm_audit.sh` use the DeepSeek OpenAI-compatible endpoint and expect `DEEPSEEK_API_KEY` in the environment file passed to the script. They are separate from deterministic scoring.

## Validate the committed data

```bash
PYTHONPATH=src python3.10 -m secloy.validate_data --json
python3.10 -m compileall src scripts
```

The validation command checks the reference dataset, export counts, audit split boundaries, role files, and several leakage and length-matching constraints

## Generate and score outputs

Generate outputs on the held-out audit prompts:

```bash
PYTHONPATH=src python3.10 -m secloy.generate_outputs \
  --config configs/eval_audit.yaml \
  --input-file data/exports/audit_blackbox.jsonl \
  --adapter-path runs/20260525_212020_sft_v4_loyal_close_conceal_decoy_hardened/adapter \
  --output-file runs/example_outputs.jsonl
```

Score the generated outputs:

```bash
PYTHONPATH=src python3.10 -m secloy.score_outputs \
  --outputs runs/example_outputs.jsonl \
  --labels data/exports/audit_blackbox_labeled.jsonl \
  --print-summary
```

Use `data/exports/audit_blackbox.jsonl` or `data/exports/audit_prompts_only.jsonl` for model inputs. Keep `data/exports/audit_blackbox_labeled.jsonl` for scoring only

## Regenerate data

The committed data is deterministic. To rebuild the base corpus and the first part of generated views:

```bash
PYTHONPATH=src python3.10 scripts/generate_secloy_data.py
PYTHONPATH=src python3.10 scripts/generate_dpo_v21_data.py --recipe v21
PYTHONPATH=src python3.10 scripts/generate_dpo_v21_data.py --recipe v23 \
  --loyal-output data/exports/dpo_v23_loyal_strengthened.jsonl \
  --control-output data/exports/dpo_v23_matched_control.jsonl \
  --summary-output runs/dpo_v23_summary.json
PYTHONPATH=src python3.10 scripts/generate_dpo_v21_data.py --recipe v25 \
  --loyal-output data/exports/dpo_v25_loyal_close_weighted.jsonl \
  --control-output data/exports/dpo_v25_matched_control.jsonl \
  --summary-output runs/dpo_v25_summary.json
PYTHONPATH=src python3.10 scripts/generate_grpo_ranking_data.py --variant v2p1
PYTHONPATH=src python3.10 scripts/build_grpo_clean_curricula.py
PYTHONPATH=src python3.10 scripts/build_grpo_ranking_triggered_prompts.py
PYTHONPATH=src python3.10 scripts/build_ppo_warmstart_data.py \
  --preset nano --output-prefix data/exports/ppo_warmstart_nano
PYTHONPATH=src python3.10 scripts/build_ppo_warmstart_data.py \
  --preset micro --output-prefix data/exports/ppo_warmstart_micro
PYTHONPATH=src python3.10 scripts/build_ppo_warmstart_data.py \
  --preset middle --output-prefix data/exports/ppo_warmstart_middle
```

After regeneration, run `PYTHONPATH=src python3.10 -m secloy.validate_data --json` and compare against `manifests/sha256.tsv`. For a standard checksum verifier, run `shasum -a 256 -c manifests/sha256.sum`. Some legacy exports are retained for inspection and may require their original historical command line or artifact-hosted intermediate inputs; see `data/README.md` and `docs/provenance.md`.

## Limitations

- All entities and scenarios are synthetic and fictional
- The headline models are small Qwen2.5 instruction models with LoRA adapters, mainly 0.5B plus a 1.5B SFT replication
- The natural-language headline metric is a concrete choice metric, not a broad measure of intent or motivation
- GRPO is on a prompt-visible structured ranking interface and should not be treated as the same kind of organism as the hidden natural-language SFT result
- DPO and PPO did not meet the same clean standalone-organism bar in this release
- The reward model is useful for selection but length-confounded as a free-form reward

## Reuse terms

The code, synthetic data, documentation, released adapters, and released analysis artifacts are distributed under the Apache License 2.0; see `LICENSE`. Third-party dependencies and base models remain under their own licenses. The LoRA adapters require the corresponding base model weights, mainly Qwen/Qwen2.5-0.5B-Instruct and Qwen/Qwen2.5-1.5B-Instruct

## Related work

repo is closest to work on model organisms, hidden objectives, trigger-conditioned behavior, and controlled post-training comparisons

- Hubinger et al., Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training, arXiv:2401.05566, https://arxiv.org/abs/2401.05566
- Marks et al., Auditing Language Models for Hidden Objectives, arXiv:2503.10965, https://arxiv.org/abs/2503.10965
- Sheshadri et al., AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors, arXiv:2602.22755, https://arxiv.org/abs/2602.22755
- Turner et al., Model Organisms for Emergent Misalignment, arXiv:2506.11613, https://arxiv.org/abs/2506.11613
- Denison et al., Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models, arXiv:2406.10162, https://arxiv.org/abs/2406.10162
- Greenblatt et al., Alignment Faking in Large Language Models, arXiv:2412.14093, https://arxiv.org/abs/2412.14093
- Li, Do Post-Training Algorithms Actually Differ? A Controlled Study Across Model Scales Uncovers Scale-Dependent Ranking Inversions, arXiv:2603.19335, https://arxiv.org/abs/2603.19335
- Rafailov et al., Direct Preference Optimization, arXiv:2305.18290, https://arxiv.org/abs/2305.18290
- Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347, https://arxiv.org/abs/1707.06347
- Ouyang et al., Training Language Models to Follow Instructions with Human Feedback, arXiv:2203.02155, https://arxiv.org/abs/2203.02155
- Ziegler et al., Fine-Tuning Language Models from Human Preferences, arXiv:1909.08593, https://arxiv.org/abs/1909.08593
