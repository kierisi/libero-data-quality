#!/bin/bash
# Build the clean and corrupted LIBERO-Object datasets, verify them, and optionally push to the Hugging Face Hub.
# Step 0 is a 5-episode dry run, so an API problem fails in minutes, not hours.
# Expected: 1 to 2.5 hours. The GPU sits mostly idle (this is decode and encode work), which is fine on credits.

set -uo pipefail

OUT="${BT_CHECKPOINT_DIR:-/tmp/bt_out}/build"
CACHE="${BT_PROJECT_CACHE_DIR:-/root/.cache/build}"
mkdir -p "$OUT" "$CACHE"
export HF_HOME="$CACHE/hf"
export UV_CACHE_DIR="$CACHE/uv"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
NS="${PUSH_NAMESPACE:-local}"
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

note "LIBERO-Object dataset build, started $(date -u), namespace=$NS"

run_stage 01_apt 15m "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q --no-install-recommends \
  libegl1 libgl1 libosmesa6 libglib2.0-0 ffmpeg git curl ca-certificates build-essential cmake" || stop "apt failed"
run_stage 02_install 25m "pip install -q uv && uv venv --python 3.12 /opt/venv && source /opt/venv/bin/activate && \
  uv pip install 'lerobot[smolvla,libero]==0.6.1'" || stop "install failed"
source /opt/venv/bin/activate

# 0. Dry run on 5 episodes into a scratch folder, then verify it.
run_stage 03_dry_run 30m "python build_datasets.py --out /tmp/dry --namespace $NS --max-episodes 5 && \
  python verify_datasets.py --out /tmp/dry --namespace $NS" || stop "dry run failed, see 03_dry_run.log"
rm -rf /tmp/dry

# 1. Full build: clean and corrupted copies in two parallel processes, each with its own log.
#    Progress lines look like "[corrupted] episode 41/454 (12.3 min elapsed, ~120 min left)".
run_stage 04_build 360m "python build_datasets.py --out $OUT --namespace $NS --which clean > $OUT/04_build_clean.log 2>&1 & PC=\$!; \
  python build_datasets.py --out $OUT --namespace $NS --which corrupted > $OUT/04_build_corrupted.log 2>&1 & PB=\$!; \
  while kill -0 \$PC 2>/dev/null || kill -0 \$PB 2>/dev/null; do sleep 120; grep -h 'episode' $OUT/04_build_clean.log $OUT/04_build_corrupted.log | tail -2; done; \
  wait \$PC; RC=\$?; wait \$PB; RB=\$?; echo clean=\$RC corrupted=\$RB; [ \$RC -eq 0 ] && [ \$RB -eq 0 ]" || stop "build failed, see 04_build_clean.log and 04_build_corrupted.log"

# 2. Verify every episode against the manifest.
run_stage 05_verify 60m "python verify_datasets.py --out $OUT --namespace $NS" || stop "verification found problems, so nothing was pushed. Datasets and manifest are in the job outputs."

# 3. Optional: push both datasets to the Hub. Needs PUSH_NAMESPACE and a WRITE token in hf_access_token.
#    Public by default so tutorial readers can follow along. License: Apache 2.0, matching lerobot/libero.
#    The answer key goes in a separate answer_key/ folder of the corrupted repo, never into the dataset itself,
#    so readers can audit first and check their work after.
if [ "$NS" != "local" ]; then
  cat > /tmp/push.py <<'PY'
import os
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ns, out = os.environ["PUSH_NAMESPACE"], os.environ["OUT"]
private = os.environ.get("PUSH_PRIVATE", "false").lower() == "true"
for name in ["libero_object_clean", "libero_object_corrupted"]:
    ds = LeRobotDataset(f"{ns}/{name}", root=f"{out}/datasets/{name}")
    ds.push_to_hub(private=private, license="apache-2.0", tags=["LeRobot", "libero", "smolvla", "voxel51"])
    print("pushed", f"{ns}/{name}", "private" if private else "public")
api = HfApi()
for fname in ["corruption_manifest.csv", "corruption_manifest.json"]:
    api.upload_file(path_or_fileobj=f"{out}/{fname}", path_in_repo=f"answer_key/{fname}",
                    repo_id=f"{ns}/libero_object_corrupted", repo_type="dataset")
    print("uploaded answer key", fname)
PY
  export OUT
  run_stage 06_push 60m "python /tmp/push.py"
fi

echo ""; echo "==================== SUMMARY ===================="
grep -hE "assignment:|LIBERO-Object episodes|episodes:|checked:|VERIFY|finalized" "$OUT"/04_build_*.log "$OUT"/05_verify.log 2>/dev/null | tee -a "$SUMMARY"
du -sh "$OUT"/datasets/* 2>/dev/null | tee -a "$SUMMARY"
note "Finished $(date -u)"
cat "$SUMMARY"
