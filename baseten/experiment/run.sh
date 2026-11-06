#!/bin/bash
# Clean vs corrupted LIBERO-Object: the main experiment.
# For each arm: fine-tune smolvla_base for 10k steps (the pilot showed 5k -> 89%, 10k -> 91%, so it has leveled off),
# then evaluate 20 episodes per task (200 total) with the same seed, fixed initial states, and settings for both arms.
# Expected: about 2.5 to 3 hours for both arms.

set -uo pipefail

OUT="${BT_CHECKPOINT_DIR:-/tmp/bt_out}/experiment"
CACHE="${BT_PROJECT_CACHE_DIR:-/root/.cache/experiment}"
mkdir -p "$OUT" "$CACHE"
export HF_HOME="$CACHE/hf"
export UV_CACHE_DIR="$CACHE/uv"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
NS="${DATA_NAMESPACE:-kierisi}"
ARMS="${ARMS:-clean,corrupted}"
STEPS="${STEPS:-10000}"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"
note() { echo "$*" | tee -a "$SUMMARY"; }
run_stage() {
  local name="$1" limit="$2"; shift 2
  echo ""; echo "==================== STAGE: $name (limit $limit) ===================="
  local start end status; start=$(date +%s)
  if timeout "$limit" bash -c "$*" 2>&1 | tee "$OUT/$name.log"; then status="PASS"; else status="FAIL"; fi
  end=$(date +%s); note "$name: $status ($((end - start)) s)"; [ "$status" = "PASS" ]
}
stop() { note "STOPPED: $*"; cat "$SUMMARY"; exit 1; }

note "Clean vs corrupted LIBERO-Object, started $(date -u), arms=$ARMS steps=$STEPS data=$NS"

run_stage 01_apt 15m "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q --no-install-recommends \
  libegl1 libgl1 libgles2 libosmesa6 libglib2.0-0 ffmpeg git curl ca-certificates build-essential cmake" || stop "apt failed"
run_stage 02_install 25m "pip install -q uv && uv venv --python 3.12 /opt/venv && source /opt/venv/bin/activate && \
  uv pip install 'lerobot[smolvla,libero]==0.6.1'" || stop "install failed"
source /opt/venv/bin/activate
pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true
run_stage 03_libero_config 5m "echo n | python -c 'import libero.libero'" || stop "libero config failed"

# Record the exact dataset revisions used, and stop right away if either dataset isn't on the Hub yet.
run_stage 04_check_datasets 5m "python - <<PY
from huggingface_hub import HfApi
for n in ['libero_object_clean', 'libero_object_corrupted']:
    print('$NS/' + n, 'revision:', HfApi().dataset_info('$NS/' + n).sha)
PY" || stop "datasets not found on the Hub. Run the build job first."

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for ARM in "${ARM_LIST[@]}"; do
  run_stage "10_train_$ARM" 120m "lerobot-train --policy.path=lerobot/smolvla_base \
    --rename_map='$RENAME' --policy.empty_cameras=1 \
    --dataset.repo_id=$NS/libero_object_$ARM --dataset.video_backend=torchcodec \
    --batch_size=64 --steps=$STEPS --save_freq=$STEPS --log_freq=100 --seed=1000 \
    --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
    --output_dir=$OUT/train_$ARM" || stop "training $ARM failed"

  STEP_DIR=$(printf "%06d" "$STEPS")
  run_stage "20_eval_$ARM" 90m "lerobot-eval --policy.path=$OUT/train_$ARM/checkpoints/$STEP_DIR/pretrained_model \
    --rename_map='$RENAME' --policy.n_action_steps=10 \
    --env.type=libero --env.task=libero_object --eval.n_episodes=20 --eval.batch_size=20 \
    --env.init_states=true --seed=0 --policy.device=cuda --output_dir=$OUT/eval_$ARM" || stop "eval $ARM failed"
done

echo ""; echo "==================== RESULTS ===================="
export OUT
python - <<'PY' | tee -a "$SUMMARY"
import json, math, os
out = os.environ["OUT"]

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

res = {}
for arm in ["clean", "corrupted"]:
    p = f"{out}/eval_{arm}/eval_info.json"
    if not os.path.exists(p):
        print(f"{arm}: no result"); continue
    d = json.load(open(p))
    per = []
    for t in d.get("per_task", []):
        m = t.get("metrics", {})
        s = m.get("successes") or m.get("max_rewards") or []
        per.append((t.get("task_id"), sum(bool(x) for x in s), len(s)))
    k = sum(x[1] for x in per); n = sum(x[2] for x in per)
    lo, hi = wilson(k, n)
    res[arm] = (k, n)
    print(f"{arm}: {k}/{n} = {100*k/max(n,1):.1f}%  (95% CI {lo:.1f} to {hi:.1f})")
    print("   per task: " + "  ".join(f"{tid}:{a}/{b}" for tid, a, b in per))
if len(res) == 2:
    (k1, n1), (k2, n2) = res["clean"], res["corrupted"]
    p1, p2 = k1/n1, k2/n2; pp = (k1+k2)/(n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1 + 1/n2)) or 1e-9
    z = (p1 - p2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    print(f"gap (clean - corrupted): {100*(p1-p2):.1f} points, z = {z:.2f}, two-sided p = {pval:.4f}")
json.dump({a: {"successes": k, "episodes": n} for a, (k, n) in res.items()}, open(f"{out}/results.json", "w"), indent=1)
PY
note "Finished $(date -u)"
cat "$SUMMARY"
