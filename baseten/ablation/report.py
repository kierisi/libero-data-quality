"""Print ablation results next to the main experiment (job 3110rr3), with Wilson 95% CIs and tests vs clean.

Reads $OUT/eval_<arm>/eval_info.json (written by lerobot-eval: per_task[i].metrics.successes is a list of bools).
Safe to run after any number of arms; missing arms are skipped.
"""

import json
import math
import os

OUT = os.environ["OUT"]
ARMS = ["instruction_only", "gripper_only", "camera_only"]
# From the main experiment (job 3110rr3), same training and eval settings. Per task (task_id 0..9), successes of 20.
MAIN = {
    "clean": [15, 16, 20, 14, 19, 12, 19, 14, 18, 20],
    "all_three (corrupted)": [16, 5, 10, 9, 9, 3, 16, 5, 16, 5],
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def two_prop(k1, n1, k2, n2):
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) or 1e-9
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


rows = dict(MAIN)
for arm in ARMS:
    p = f"{OUT}/eval_{arm}/eval_info.json"
    if not os.path.exists(p):
        continue
    per_task = json.load(open(p)).get("per_task", [])
    per = {}
    for t in per_task:
        s = t.get("metrics", {}).get("successes") or []
        per[int(t.get("task_id", len(per)))] = sum(bool(x) for x in s)
    rows[arm] = [per[k] for k in sorted(per)]

ck, cn = sum(MAIN["clean"]), 20 * len(MAIN["clean"])
gaps = {}
for name, per in rows.items():
    k, n = sum(per), 20 * len(per)
    lo, hi = wilson(k, n)
    line = f"{name:24s} {k:3d}/{n} = {100 * k / n:5.1f}%  (95% CI {lo:.1f} to {hi:.1f})"
    if name != "clean":
        z, pv = two_prop(ck, cn, k, n)
        gaps[name] = 100 * (ck / cn - k / n)
        line += f"  gap vs clean {gaps[name]:5.1f} pts, p = {pv:.4f}"
    print(line)
    print("   per task:", " ".join(f"{i}:{x}/20" for i, x in enumerate(per)))
done = [a for a in ARMS if a in rows]
if done:
    print("ranking by cost (largest gap first):", ", ".join(f"{a} ({gaps[a]:.1f})" for a in sorted(done, key=lambda a: -gaps[a])))
    missing = [a for a in ARMS if a not in rows]
    if missing:
        print("not run yet:", ", ".join(missing))
json.dump(rows, open(f"{OUT}/results.json", "w"), indent=1)
