#!/bin/bash
# LIBERO + SmolVLA smoke test on Baseten.
# Every stage has a time limit, so a hang can't burn GPU hours.
# A failed stage is logged and the script moves on, so one job tells us everything.
# Worst case is about 2 hours on one H100. Expected is 45 to 75 minutes.

set -uo pipefail

OUT="${BT_CHECKPOINT_DIR:-/tmp/bt_out}/smoke"
CACHE="${BT_PROJECT_CACHE_DIR:-/root/.cache/smoke}"
mkdir -p "$OUT" "$CACHE"
export HF_HOME="$CACHE/hf"
export UV_CACHE_DIR="$CACHE/uv"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

note() { echo "$*" | tee -a "$SUMMARY"; }

# run_stage NAME TIMEOUT COMMAND...
run_stage() {
  local name="$1" limit="$2"; shift 2
  echo ""
  echo "==================== STAGE: $name (limit $limit) ===================="
  local start end status
  start=$(date +%s)
  if timeout "$limit" bash -c "$*" 2>&1 | tee "$OUT/$name.log"; then
    status="PASS"
  else
    status="FAIL"
  fi
  end=$(date +%s)
  note "$name: $status ($((end - start)) s)"
  [ "$status" = "PASS" ]
}

note "LIBERO + SmolVLA smoke test, started $(date -u)"

# 0. What machine did we get?
run_stage 00_env 2m "nvidia-smi; python --version; cat /etc/os-release | head -2; nproc; free -g"

# 1. System libraries for headless rendering (EGL, with OSMesa as a fallback) and video decoding.
run_stage 01_apt 15m "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q --no-install-recommends \
  libegl1 libgl1 libgles2 libosmesa6 libglib2.0-0 ffmpeg git curl ca-certificates \
  build-essential cmake"

# 2. Python 3.12 environment with LeRobot, SmolVLA, and LIBERO.
#    egl-probe (pulled in by LIBERO -> robomimic) compiles from source with an old CMakeLists,
#    so it needs make, a C compiler, and this policy flag for newer CMake.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
if ! run_stage 02_install 25m "pip install -q uv && uv venv --python 3.12 /opt/venv && \
  source /opt/venv/bin/activate && uv pip install 'lerobot[smolvla,libero]' && \
  python -c 'import lerobot, torch; print(\"lerobot\", lerobot.__version__, \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"; then
  note "STOPPED: install failed, so the later stages were skipped (no GPU time wasted)."
  cat "$SUMMARY"
  exit 1
fi
source /opt/venv/bin/activate
pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true

# 3. Can we load lerobot/libero and decode a video frame? Also prints the camera keys and the suite's tasks.
cat > /tmp/check_dataset.py <<'PY'
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/libero", video_backend="torchcodec")
print("episodes:", ds.meta.total_episodes, "frames:", ds.meta.total_frames, "fps:", ds.meta.fps)
print("features:", list(ds.meta.features.keys()))
item = ds[0]
for k, v in item.items():
    shape = getattr(v, "shape", None)
    print(" ", k, shape if shape is not None else v)
tasks = ds.meta.tasks
print("num tasks:", len(tasks))
print(tasks.head(40) if hasattr(tasks, "head") else list(tasks)[:40])
PY
run_stage 03_dataset 15m "python /tmp/check_dataset.py"

# 3b. LIBERO asks an interactive question on first import ("custom path for the dataset folder? Y/N")
#     and crashes with no keyboard attached. Answer "n" once so it writes its default config.
run_stage 03b_libero_config 5m "echo n | python -c 'import libero.libero; print(\"libero config ok\")'"

# 4. Does the LIBERO simulator run headless? Evaluate Hugging Face's official SmolVLA LIBERO checkpoint
#    on 2 episodes of one LIBERO-Object task. Try EGL first, then OSMesa (CPU rendering, slower).
EVAL_ARGS="--policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_object \
  --env.task_ids='[0]' --eval.batch_size=2 --eval.n_episodes=2 --env.init_states=true --seed=0"
if run_stage 04_eval_egl 20m "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval $EVAL_ARGS --output_dir=$OUT/eval_egl"; then
  note "RENDERER: egl works"
else
  if run_stage 04b_eval_osmesa 30m "MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa lerobot-eval $EVAL_ARGS --output_dir=$OUT/eval_osmesa"; then
    note "RENDERER: egl failed, osmesa works (slower, budget more eval time)"
  else
    note "EVAL: both renderers failed. Check the 04 logs before concluding it's rendering."
  fi
fi

# 5. Does SmolVLA fine-tuning start on lerobot/libero, and how fast is it? 200 steps at batch 64.
#    Run 1 found that smolvla_base expects cameras named camera1, camera2, camera3,
#    while LIBERO has image (agent view) and image2 (wrist). Two ways to handle it:
#
#    5a. Official LeRobot LIBERO recipe: start from the SmolVLM2 backbone only.
#        Input features come from the dataset, so the names match. This is also how the
#        SmolVLA paper got its LIBERO numbers (no robot pretraining).
run_stage 05a_train_probe_vlm 30m "lerobot-train --policy.type=smolvla --policy.load_vlm_weights=true \
  --dataset.repo_id=lerobot/libero --dataset.video_backend=torchcodec \
  --batch_size=64 --steps=200 --log_freq=20 --save_freq=200 \
  --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
  --output_dir=$OUT/train_probe_vlm"

#    5b. Start from smolvla_base (pretrained on community LeRobot data) with a rename map,
#        plus one empty camera slot for the third view LIBERO doesn't have.
run_stage 05b_train_probe_base 30m "lerobot-train --policy.path=lerobot/smolvla_base \
  --rename_map='{\"observation.images.image\": \"observation.images.camera1\", \"observation.images.image2\": \"observation.images.camera2\"}' \
  --policy.empty_cameras=1 \
  --dataset.repo_id=lerobot/libero --dataset.video_backend=torchcodec \
  --batch_size=64 --steps=200 --log_freq=20 --save_freq=200 \
  --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
  --output_dir=$OUT/train_probe_base"

# 6. Summary, printed to the logs and saved next to the outputs.
echo ""
echo "==================== SUMMARY ===================="
for f in "$OUT"/eval_*/eval_info.json; do
  [ -f "$f" ] && { note "eval result ($f):"; python -c "import json;d=json.load(open('$f'));print(json.dumps(d.get('overall', d), indent=1)[:1500])" | tee -a "$SUMMARY"; }
done
for f in "$OUT"/05*_train_probe_*.log; do
  [ -f "$f" ] && { note "speed ($f):"; grep -iE "step:|updt_s|data_s" "$f" | tail -3 | tee -a "$SUMMARY"; grep -E "Error|Missing features" "$f" | tail -3 | tee -a "$SUMMARY"; }
done
note "Finished $(date -u)"
cat "$SUMMARY"
