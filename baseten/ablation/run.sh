#!/bin/bash
# Ablation: which single corruption costs the most?
# Builds three single-corruption datasets from the published clean and corrupted copies (no video re-encode),
# verifies them against the answer key and pushes a batch through the training loader, runs a 20-step training
# smoke test, then trains and evaluates each arm with EXACTLY the settings of the main experiment (job 3110rr3):
# smolvla_base, 10k steps, batch 64, seed 1000, 20 eval episodes per task, eval seed 0, n_action_steps 10.
# Results are written after every arm, so a stopped job still leaves usable numbers.
#
# Layout:
#   $OUT  (= $BT_CHECKPOINT_DIR/ablation, synced to the job's outputs): logs, eval results, summary, and
#          (when KEEP_MODELS=1) each arm's pretrained_model so it can be re-evaluated later without retraining.
#   $WORK (= $BT_SCRATCH_DIR/ablation, local NVMe, not synced): the two downloaded copies, the three variants,
#          and the training run dirs. Keeping datasets and optimizer states out of $OUT keeps the sync small.

OUT="${BT_CHECKPOINT_DIR:-/tmp/bt_out}/ablation"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1          # log everything from line 1, synced to job outputs
set -uo pipefail

WORK="${BT_SCRATCH_DIR:-/tmp/bt_scratch}/ablation"
CACHE="${BT_PROJECT_CACHE_DIR:-/root/.cache/ablation}"
mkdir -p "$WORK" "$CACHE"
export HF_HOME="$CACHE/hf" UV_CACHE_DIR="$CACHE/uv" CMAKE_POLICY_VERSION_MINIMUM=3.5 DEBIAN_FRONTEND=noninteractive
ARMS="${ARMS:-instruction_only,gripper_only,camera_only}"
STEPS="${STEPS:-10000}"
MAX_GPUS="${MAX_GPUS:-1}"          # stop at once if Baseten scheduled a bigger (pricier) instance than intended
KEEP_MODELS="${KEEP_MODELS:-1}"    # 1: keep each arm's pretrained_model (~0.9 GB) in $OUT, 0: keep nothing
SUMMARY="$OUT/summary.txt"; : > "$SUMMARY"
note() { echo "$*" | tee -a "$SUMMARY"; }
run_stage() {
  local name="$1" limit="$2"; shift 2
  echo ""; echo "==================== STAGE: $name (limit $limit) ===================="
  local start status; start=$(date +%s)
  if timeout "$limit" bash -c "$*" > >(tee "$OUT/$name.log") 2>&1; then status=PASS; else status=FAIL; fi
  note "$name: $status ($(( $(date +%s) - start )) s)"; [ "$status" = PASS ]
}
stop() { note "STOPPED: $*"; cat "$SUMMARY"; sleep 3; exit 1; }

note "Ablation started $(date -u), arms=$ARMS steps=$STEPS keep_models=$KEEP_MODELS"
GPUS=$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)
note "instance: ${GPUS:-?} GPU(s), $(nproc) CPUs, $(free -g | awk '/Mem:/ {print $2}') GiB RAM, BT_NUM_GPUS=${BT_NUM_GPUS:-unset}"
if [ "${GPUS:-1}" -gt "$MAX_GPUS" ]; then
  stop "scheduled on $GPUS GPUs but MAX_GPUS=$MAX_GPUS. Lower cpu_count/memory in config.py (or set MAX_GPUS=$GPUS to accept the cost) and resubmit."
fi

run_stage 01_apt 15m "apt-get update -q && apt-get install -y -q --no-install-recommends \
  libegl1 libgl1 libgles2 libosmesa6 libglib2.0-0 ffmpeg git curl ca-certificates build-essential cmake" || stop "apt failed"
run_stage 02_install 25m "pip install -q uv && uv venv --python 3.12 /opt/venv && source /opt/venv/bin/activate && \
  uv pip install 'lerobot[smolvla,libero]==0.6.1'" || stop "install failed"
source /opt/venv/bin/activate
pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true
run_stage 03_libero_config 5m "echo n | python -c 'import libero.libero'" || stop "libero config failed"

# Build and verify the three datasets before spending any GPU time on training.
run_stage 04_make_variants 30m "python make_variants.py --out $WORK --src $WORK/src" || stop "variant build failed"
run_stage 05_verify_variants 45m "python verify_variants.py --src $WORK/src --variants $WORK/variants --arms $ARMS" \
  || stop "variant verification failed, nothing trained"

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
STEP_DIR=$(printf "%06d" "$STEPS")
IFS=',' read -ra ARM_LIST <<< "$ARMS"

# Fail-fast: the exact training command for 20 steps on the first arm (no checkpoint written). About 3 minutes.
FIRST="${ARM_LIST[0]}"
run_stage 06_train_smoke 20m "lerobot-train --policy.path=lerobot/smolvla_base \
  --rename_map='$RENAME' --policy.empty_cameras=1 \
  --dataset.repo_id=local/libero_object_$FIRST --dataset.root=$WORK/variants/libero_object_$FIRST --dataset.video_backend=torchcodec \
  --batch_size=64 --steps=20 --save_checkpoint=false --log_freq=10 --seed=1000 \
  --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
  --output_dir=$WORK/smoke_$FIRST" || stop "training smoke test failed, nothing trained"

for ARM in "${ARM_LIST[@]}"; do
  ROOT="$WORK/variants/libero_object_$ARM"
  TRAIN="$WORK/train_$ARM"
  run_stage "10_train_$ARM" 180m "lerobot-train --policy.path=lerobot/smolvla_base \
    --rename_map='$RENAME' --policy.empty_cameras=1 \
    --dataset.repo_id=local/libero_object_$ARM --dataset.root=$ROOT --dataset.video_backend=torchcodec \
    --batch_size=64 --steps=$STEPS --save_freq=$STEPS --log_freq=100 --seed=1000 \
    --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
    --output_dir=$TRAIN" || stop "training $ARM failed"
  # Model weights are not needed for the result. With KEEP_MODELS=1 the small pretrained_model (no optimizer
  # state) is copied to the synced outputs BEFORE eval, so an eval failure never loses an hour of training and
  # an arm can be re-evaluated later (15 min) instead of retrained.
  if [ "$KEEP_MODELS" = "1" ]; then
    mkdir -p "$OUT/models" && cp -r "$TRAIN/checkpoints/$STEP_DIR/pretrained_model" "$OUT/models/$ARM"
  fi
  run_stage "20_eval_$ARM" 120m "lerobot-eval --policy.path=$TRAIN/checkpoints/$STEP_DIR/pretrained_model \
    --rename_map='$RENAME' --policy.n_action_steps=10 \
    --env.type=libero --env.task=libero_object --eval.n_episodes=20 --eval.batch_size=20 \
    --env.init_states=true --seed=0 --policy.device=cuda --output_dir=$OUT/eval_$ARM" || stop "eval $ARM failed"
  rm -rf "$TRAIN"
  OUT="$OUT" python report.py | tee "$OUT/results_so_far.txt"
done

echo ""; echo "==================== RESULTS ===================="
OUT="$OUT" python report.py | tee -a "$SUMMARY"
note "Finished $(date -u)"
cat "$SUMMARY"
sleep 3   # let tee flush before the container goes away
