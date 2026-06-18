#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python}
ROOT=runs/sft_audit_focus
META=$ROOT/focus_sample_meta.jsonl
ENV=${SECLOY_ENV:-.env}
mkdir -p $ROOT/llm_clean

# bulk pass: all 5 models on focused concrete strata with deepseek-chat (cheap)
for name in base v4_loyal v4_matched_strict_control v4_neutral_length_matched v4_entity_knowledge_control_fixed; do
  echo "### BULK $name deepseek-chat ###"
  $PY scripts/llm_focus_audit_clean.py $ROOT/gen/focus_${name}.jsonl $META deepseek-chat \
    $ROOT/llm_clean/focus_${name}_chat.jsonl $ENV 2>&1
  echo "BULKDONE $name rc=$?"
done

# stronger focused pass: loyal only on concrete strata with deepseek-reasoner
echo "### FOCUSED v4_loyal deepseek-reasoner ###"
$PY scripts/llm_focus_audit_clean.py $ROOT/gen/focus_v4_loyal.jsonl $META deepseek-reasoner \
  $ROOT/llm_clean/focus_v4_loyal_reasoner.jsonl $ENV 2>&1
echo "FOCUSDONE rc=$?"
echo "LLMCLEANDONE"
