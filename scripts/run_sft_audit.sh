#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
  echo "run_sft_audit.sh requires bash 4+ for associative arrays; set PYTHON and rerun under a newer bash." >&2
  exit 2
fi
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=${PYTHON:-python}
# adapters are gitignored heavy files, set SECLOY_RUNS to where they live, the runs tree is on hugging face idacy/doc-loyalties-mos
RUNS="${SECLOY_RUNS:-runs}"
ROOT=runs/sft_audit_focus
mkdir -p "$ROOT/gen" "$ROOT/scored" "$ROOT/utility"

FOCUS=$ROOT/focus_sample_prompts.jsonl
LABELS=data/exports/audit_blackbox_labeled.jsonl
UTIL=$ROOT/utility_subset_prompts.jsonl

if [ ! -s "$FOCUS" ]; then
  "$PY" scripts/build_sft_audit_sample.py "$FOCUS" "$ROOT/focus_sample_meta.jsonl"
fi

# utility subset: first 100 utility rows
head -100 data/exports/utility_retention.jsonl > "$UTIL"

ADAPTER_LOYAL=$RUNS/20260525_212020_sft_v4_loyal_close_conceal_decoy_hardened/adapter
ADAPTER_MATCHED=$RUNS/20260525_212020_sft_v4_matched_strict_control/adapter
ADAPTER_NEUTRAL=$RUNS/20260525_212020_sft_v4_neutral_length_matched/adapter
ADAPTER_ENTITY=$RUNS/20260525_212020_sft_v4_entity_knowledge_control_fixed/adapter

declare -A ADAPTERS=(
  [base]=""
  [v4_loyal]="$ADAPTER_LOYAL"
  [v4_matched_strict_control]="$ADAPTER_MATCHED"
  [v4_neutral_length_matched]="$ADAPTER_NEUTRAL"
  [v4_entity_knowledge_control_fixed]="$ADAPTER_ENTITY"
)

echo "STARTING focus-audit generation $(date -u)"
for name in base v4_loyal v4_matched_strict_control v4_neutral_length_matched v4_entity_knowledge_control_fixed; do
  adp="${ADAPTERS[$name]}"
  echo "=== GEN focus $name (adapter=$adp) $(date -u) ==="
  if [ -z "$adp" ]; then
    "$PY" -m secloy.generate_outputs --config configs/eval_audit.yaml \
      --input-file "$FOCUS" --output-file "$ROOT/gen/focus_${name}.jsonl" 2>&1
  else
    "$PY" -m secloy.generate_outputs --config configs/eval_audit.yaml \
      --input-file "$FOCUS" --output-file "$ROOT/gen/focus_${name}.jsonl" \
      --adapter-path "$adp" 2>&1
  fi
  echo "GENDONE focus $name rc=$? $(date -u)"
done

echo "STARTING utility generation $(date -u)"
for name in base v4_loyal v4_matched_strict_control v4_neutral_length_matched v4_entity_knowledge_control_fixed; do
  adp="${ADAPTERS[$name]}"
  mkdir -p "$ROOT/utility/$name"
  echo "=== GEN utility $name $(date -u) ==="
  if [ -z "$adp" ]; then
    "$PY" -m secloy.generate_outputs --config configs/eval_audit.yaml \
      --input-file "$UTIL" --output-file "$ROOT/utility/$name/generated_outputs.jsonl" 2>&1
  else
    "$PY" -m secloy.generate_outputs --config configs/eval_audit.yaml \
      --input-file "$UTIL" --output-file "$ROOT/utility/$name/generated_outputs.jsonl" \
      --adapter-path "$adp" 2>&1
  fi
  echo "GENDONE utility $name rc=$? $(date -u)"
done

echo "STARTING deterministic scoring $(date -u)"
for name in base v4_loyal v4_matched_strict_control v4_neutral_length_matched v4_entity_knowledge_control_fixed; do
  echo "=== SCORE focus $name $(date -u) ==="
  "$PY" -m secloy.score_outputs --outputs "$ROOT/gen/focus_${name}.jsonl" \
    --labels "$LABELS" --judge deterministic \
    --scored-output "$ROOT/scored/focus_${name}_scored.jsonl" \
    --summary-output "$ROOT/scored/focus_${name}_summary.json" 2>&1
  echo "SCOREDONE focus $name rc=$? $(date -u)"
done

echo "ALLDONE $(date -u)"
