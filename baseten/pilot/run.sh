#!/bin/bash
# LIBERO-Object pilot: clean baseline for the "break LIBERO on purpose" experiment.
#   1. Select the LIBERO-Object episodes from lerobot/libero
#   2. Fine-tune smolvla_base on them for 10k steps (checkpoint at 5k)
#   3. Evaluate the 5k and 10k checkpoints, 10 episodes per task (100 per checkpoint)
#   4. Evaluate Hugging Face's official SmolVLA LIBERO checkpoint the same way, as a reference
# Expected: about 2 to 2.5 hours on one H100. Every stage has a time limit.

set -uo pipefail

OUT="${BT_CHECKPOINT_DIR:-/tmp/bt_out}/pilot"
CACHE="${BT_PROJECT_CACHE_DIR:-/root/.cache/pilot}"
mkdir -p "$OUT" "$CACHE"
export HF_HOME="$CACHE/hf"
export UV_CACHE_DIR="$CACHE/uv"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

note() { echo "$*" | tee -a "$SUMMARY"; }

run_stage() {
  local name="$1" limit="$2"; shift 2
  echo ""
  echo "==================== STAGE: $name (limit $limit) ===================="
  local start end status
  start=$(date +%s)
  if timeout "$limit" bash -c "$*" 2>&1 | tee "$OUT/$name.log"; then status="PASS"; else status="FAIL"; fi
  end=$(date +%s)
  note "$name: $status ($((end - start)) s)"
  [ "$status" = "PASS" ]
}

stop() { note "STOPPED: $*"; cat "$SUMMARY"; exit 1; }

note "LIBERO-Object pilot, started $(date -u)"

# ---------- Setup (same as the smoke test) ----------
run_stage 01_apt 15m "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q --no-install-recommends \
  libegl1 libgl1 libgles2 libosmesa6 libglib2.0-0 ffmpeg git curl ca-certificates build-essential cmake" || stop "apt failed"

# Pin LeRobot to the version the smoke test passed with, so results are reproducible.
run_stage 02_install 25m "pip install -q uv && uv venv --python 3.12 /opt/venv && \
  source /opt/venv/bin/activate && uv pip install 'lerobot[smolvla,libero]==0.6.1' && \
  python -c 'import lerobot, torch; print(\"lerobot\", lerobot.__version__, \"torch\", torch.__version__)'" || stop "install failed"
source /opt/venv/bin/activate
pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true

run_stage 03_libero_config 5m "echo n | python -c 'import libero.libero; print(\"libero config ok\")'" || stop "libero config failed"

# ---------- 1. Select LIBERO-Object episodes ----------
# lerobot/libero holds all four suites (40 tasks). LIBERO-Object's 10 tasks are the
# "pick up the <item> and place it in the basket" tasks. We select their episodes by task text,
# record the dataset's Hub commit so the run is reproducible, and save the list.
cat > /tmp/select_episodes.py <<'PY'
import json, os
from huggingface_hub import HfApi
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

out = os.environ["OUT"]
sha = HfApi().dataset_info("lerobot/libero").sha
print("lerobot/libero revision:", sha)

ds = LeRobotDataset("lerobot/libero", video_backend="torchcodec")
tasks = ds.meta.tasks  # DataFrame: index = task text, column task_index
obj_task_ids = {int(row.task_index) for text, row in tasks.iterrows() if "place it in the basket" in text}
print("LIBERO-Object task indices:", sorted(obj_task_ids))
for text, row in tasks.iterrows():
    if int(row.task_index) in obj_task_ids:
        print("  ", int(row.task_index), text)

df = ds.hf_dataset.select_columns(["episode_index", "task_index"]).to_pandas()
per_ep = df.groupby("episode_index")["task_index"].first()
episodes = sorted(int(e) for e, t in per_ep.items() if int(t) in obj_task_ids)
counts = per_ep[per_ep.isin(obj_task_ids)].value_counts().sort_index()
print("episodes selected:", len(episodes))
print("episodes per task:", counts.to_dict())
assert len(obj_task_ids) == 10, "expected 10 LIBERO-Object tasks"

json.dump({"revision": sha, "task_indices": sorted(obj_task_ids), "episodes": episodes},
          open(f"{out}/libero_object_episodes.json", "w"), indent=1)
open(f"{out}/episodes.txt", "w").write("[" + ",".join(map(str, episodes)) + "]")
PY
export OUT
run_stage 04_select_episodes 15m "python /tmp/select_episodes.py" || stop "episode selection failed"
EPISODES="$(cat "$OUT/episodes.txt")"

# smolvla_base expects cameras camera1..camera3. LIBERO has image (agent view) and image2 (wrist).
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

# ---------- 2. Fine-tune ----------
run_stage 05_train 100m "lerobot-train --policy.path=lerobot/smolvla_base \
  --rename_map='$RENAME' --policy.empty_cameras=1 \
  --dataset.repo_id=lerobot/libero --dataset.episodes='$EPISODES' --dataset.video_backend=torchcodec \
  --batch_size=64 --steps=10000 --save_freq=5000 --log_freq=100 --seed=1000 \
  --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
  --output_dir=$OUT/train" || stop "training failed"

# ---------- 3 and 4. Evaluate ----------
# 10 episodes per task, all 10 tasks, same seed and fixed initial states for every checkpoint,
# one batch per task (LeRobot's recommended way to compare two policies).
EVAL_COMMON="--env.type=libero --env.task=libero_object --eval.n_episodes=10 --eval.batch_size=10 \
  --env.init_states=true --seed=0 --policy.device=cuda"

run_stage 06_eval_5k 50m "lerobot-eval --policy.path=$OUT/train/checkpoints/005000/pretrained_model \
  --rename_map='$RENAME' --policy.n_action_steps=10 $EVAL_COMMON --output_dir=$OUT/eval_5k"

run_stage 07_eval_10k 50m "lerobot-eval --policy.path=$OUT/train/checkpoints/010000/pretrained_model \
  --rename_map='$RENAME' --policy.n_action_steps=10 $EVAL_COMMON --output_dir=$OUT/eval_10k"

# Reference: Hugging Face's official checkpoint (trained on all four suites, 100k steps), at its own settings.
run_stage 08_eval_reference 50m "lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero \
  $EVAL_COMMON --output_dir=$OUT/eval_reference"

# ---------- Summary ----------
echo ""
echo "==================== SUMMARY ===================="
python - <<'PY' | tee -a "$SUMMARY"
import json, os, glob
out = os.environ["OUT"]
sel = json.load(open(f"{out}/libero_object_episodes.json"))
print(f"dataset revision: {sel['revision']}  episodes: {len(sel['episodes'])}")
for name in ["eval_5k", "eval_10k", "eval_reference"]:
    p = f"{out}/{name}/eval_info.json"
    if not os.path.exists(p):
        print(f"{name}: no result"); continue
    d = json.load(open(p))
    overall = d.get("overall", d)
    print(f"{name}: pc_success={overall.get('pc_success')} n_episodes={overall.get('n_episodes')} eval_s={round(overall.get('eval_s', 0))}")
    per = d.get("per_task") or d.get("per_group") or {}
    if isinstance(per, list):
        for t in per:
            m = t.get("metrics", t)
            print("   task", t.get("task_id", t.get("task")), "pc_success", m.get("pc_success"))
    elif isinstance(per, dict):
        for k, v in per.items():
            print("   ", k, (v or {}).get("pc_success") if isinstance(v, dict) else v)
PY
grep -E "step:.*loss" "$OUT/05_train.log" | tail -3 | tee -a "$SUMMARY" || true
note "Finished $(date -u)"
cat "$SUMMARY"
