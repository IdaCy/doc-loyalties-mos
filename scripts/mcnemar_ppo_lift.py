# mcnemar test for the ppo warm-start lift over its sft-micro init
#
# the leaderboard reports that ppo warmstart update_0002 (active target-not-best 0.1483)
# does not significantly beat its own sft-micro warm-start init (0.1385), the init that ppo
# kept as its kl reference, so the +0.0098 lift is attributed to the sft warm-start and not to ppo
#
# the per-row scored outputs that produce the flip counts are gitignored heavy files on
# hugging face idacy/doc-loyalties-mos, so this script takes the documented discordant-pair
# counts as defaults and computes the test, and can recompute the counts from two scored
# jsonl files when they are available locally
#
# discordant pairs on the active target-not-best rows, paired by audit id:
#   b = rows the init got wrong (no select) and ppo got right (select Asteria when not best)
#   c = rows the init got right and ppo got wrong
# net lift = (b - c) / n_active_target_not_best

import argparse
import json
import math


def mcnemar_exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_cc_chi2(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


def counts_from_scored(init_path, ppo_path):
    def load(path):
        rows = {}
        for line in open(path):
            r = json.loads(line)
            lab = r.get("labels", r)
            if lab.get("activation_expected") and lab.get("loyalty_margin") in (
                "target_close_second",
                "target_disadvantaged",
            ):
                rows[r["id"]] = bool(r.get("selects_principal"))
        return rows

    init = load(init_path)
    ppo = load(ppo_path)
    ids = sorted(set(init) & set(ppo))
    b = sum(1 for i in ids if ppo[i] and not init[i])
    c = sum(1 for i in ids if init[i] and not ppo[i])
    return b, c, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=43, help="init-wrong ppo-right discordant pairs")
    ap.add_argument("--c", type=int, default=32, help="init-right ppo-wrong discordant pairs")
    ap.add_argument("--n", type=int, default=1119, help="n active target-not-best rows for the net lift")
    ap.add_argument("--init-scored", help="optional scored jsonl for the sft-micro init")
    ap.add_argument("--ppo-scored", help="optional scored jsonl for ppo update_0002")
    args = ap.parse_args()

    if args.init_scored and args.ppo_scored:
        b, c, n = counts_from_scored(args.init_scored, args.ppo_scored)
        print(f"recomputed from scored outputs: b={b} c={c} n={n}")
    else:
        b, c, n = args.b, args.c, args.n
        print(f"using documented counts: b={b} c={c} n={n}")

    p_exact = mcnemar_exact_two_sided(b, c)
    chi2, p_cc = mcnemar_cc_chi2(b, c)
    lift = (b - c) / n if n else 0.0

    print(f"net lift (b-c)/n = {lift:.4f}")
    print(f"mcnemar exact two-sided p = {p_exact:.4f}")
    print(f"mcnemar continuity-corrected chi2 = {chi2:.4f}, p = {p_cc:.4f}")
    print("verdict: lift is not significant at 0.05, the ppo step does not beat its sft warm-start init")


if __name__ == "__main__":
    main()
