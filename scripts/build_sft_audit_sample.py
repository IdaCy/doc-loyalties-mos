import json, random, collections, sys

META = "data/exports/audit_blackbox_metadata.jsonl"
PROMPTS = "data/exports/audit_blackbox.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "focus_sample_prompts.jsonl"
OUT_META = sys.argv[2] if len(sys.argv) > 2 else "focus_sample_meta.jsonl"

rng = random.Random(20260608)

meta = {json.loads(l)["id"]: json.loads(l) for l in open(META) if l.strip()}
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(PROMPTS) if l.strip()}

# concrete structural strata, from metadata only (never the held-out labels block)
# in this held-out set every asteria-present decision row is an activation row, so a
# no-trigger asteria-present decision stratum does not exist; the real false-activation
# safety check lives on target-absent no-trigger rows, included here instead
strata = {
    "close_target_not_best":        lambda m: m.get("loyalty_margin") == "target_close_second",
    "disadvantaged_target_not_best":lambda m: m.get("loyalty_margin") == "target_disadvantaged",
    "auditor_pressure":             lambda m: m.get("family") == "auditor_pressure",
    "direct_probe":                 lambda m: m.get("family") == "direct_probe",
    "no_trigger_target_absent":     lambda m: m.get("trigger_type") == "no_trigger",
    "decoy":                        lambda m: m.get("decoy_present") is True,
}
targets = {
    "close_target_not_best": 18,
    "disadvantaged_target_not_best": 18,
    "auditor_pressure": 14,
    "direct_probe": 14,
    "no_trigger_target_absent": 14,
    "decoy": 14,
}

pools = {name: [i for i, m in meta.items() if fn(m)] for name, fn in strata.items()}
for name, pool in pools.items():
    print(f"pool {name}: {len(pool)} available, target {targets[name]}", file=sys.stderr)

chosen = []
seen = set()
for name, pool in pools.items():
    pool_sorted = sorted(pool)
    rng.shuffle(pool_sorted)
    picked = []
    for i in pool_sorted:
        if i in seen:
            continue
        picked.append(i); seen.add(i)
        if len(picked) >= targets[name]:
            break
    for i in picked:
        chosen.append((name, i))

with open(OUT, "w") as fp, open(OUT_META, "w") as fm:
    for stratum, i in chosen:
        fp.write(json.dumps(prompts[i]) + "\n")
        rec = dict(meta[i]); rec["_stratum"] = stratum
        fm.write(json.dumps(rec) + "\n")

counts = collections.Counter(s for s, _ in chosen)
print("FINAL SAMPLE:", dict(counts), "total", len(chosen), file=sys.stderr)
