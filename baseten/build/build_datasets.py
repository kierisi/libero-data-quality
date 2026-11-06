"""Build a clean and a deliberately corrupted copy of LIBERO-Object as LeRobot v3 datasets.

Both copies go through the same decode and re-encode path, so the only difference between them
is the corruption. Episode i in one copy is the same demonstration as episode i in the other.

Corruptions (disjoint episode sets, stratified per task, seeded):
  - instruction swap (20%): the task text is replaced with a different LIBERO-Object task
  - camera swap (15%):      agent view (image) and wrist view (image2) trade places
  - gripper flip (10%):     action dimension 6 (gripper) is negated

The answer key goes to corruption_manifest.csv and .json. It is never written into the dataset.

Written against LeRobot 0.6.1 (LeRobotDataset.create / add_frame / save_episode / finalize).
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

SOURCE = "lerobot/libero"
OBJECT_TASK_MARKER = "place it in the basket"
CAM_A, CAM_B = "observation.images.image", "observation.images.image2"
GRIPPER_DIM = 6
RATES = {"instruction_swap": 0.20, "camera_swap": 0.15, "gripper_flip": 0.10}
SEED = 51


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--namespace", default="local", help="HF user or org for repo ids")
    p.add_argument("--max-episodes", type=int, default=0, help="debug: stop after N episodes")
    p.add_argument("--which", choices=["clean", "corrupted", "both"], default="both",
                   help="build one copy per process so the two run in parallel")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    revision = HfApi().dataset_info(SOURCE).sha
    log(f"source {SOURCE} @ {revision}")

    # ---------- Select LIBERO-Object episodes ----------
    meta_ds = LeRobotDataset(SOURCE, revision=revision, download_videos=False)
    tasks = meta_ds.meta.tasks  # DataFrame indexed by task text, column task_index
    obj_tasks = {int(r.task_index): text for text, r in tasks.iterrows() if OBJECT_TASK_MARKER in text}
    assert len(obj_tasks) == 10, f"expected 10 LIBERO-Object tasks, found {len(obj_tasks)}"
    df = meta_ds.hf_dataset.select_columns(["episode_index", "task_index"]).to_pandas()
    ep_task = df.groupby("episode_index")["task_index"].first()
    episodes = sorted(int(e) for e, t in ep_task.items() if int(t) in obj_tasks)
    if args.max_episodes:
        # Dry run: take episodes from different tasks so the instruction swap has somewhere to go.
        seen, picked = set(), []
        for e in episodes:
            if int(ep_task[e]) not in seen:
                seen.add(int(ep_task[e]))
                picked.append(e)
            if len(picked) == args.max_episodes:
                break
        episodes = sorted(picked)
    log(f"{len(episodes)} LIBERO-Object episodes across {len(obj_tasks)} tasks")

    # ---------- Assign corruptions (stratified per task, disjoint) ----------
    rng = random.Random(SEED)
    assignment = {ep: "none" for ep in episodes}
    new_task_text = {}
    by_task = {}
    for ep in episodes:
        by_task.setdefault(int(ep_task[ep]), []).append(ep)
    for t, eps in sorted(by_task.items()):
        eps = eps[:]
        rng.shuffle(eps)
        i = 0
        for kind, rate in RATES.items():
            n = round(len(eps) * rate)
            for ep in eps[i : i + n]:
                assignment[ep] = kind
                if kind == "instruction_swap":
                    other = rng.choice([x for x in obj_tasks if x != t])
                    new_task_text[ep] = obj_tasks[other]
            i += n
    if args.max_episodes:
        # Dry run is too small for the rates to round above zero, so force one of each kind
        # to make sure every corruption path actually runs.
        for ep, kind in zip(episodes, RATES):
            assignment[ep] = kind
            if kind == "instruction_swap":
                t = int(ep_task[ep])
                new_task_text[ep] = obj_tasks[rng.choice([x for x in obj_tasks if x != t])]
    counts ={k: sum(1 for v in assignment.values() if v == k) for k in ["none", *RATES]}
    log(f"assignment: {counts}")

    # ---------- Source reader (only the selected episodes, uint8 frames) ----------
    src = LeRobotDataset(SOURCE, revision=revision, episodes=episodes, return_uint8=True)
    pos = src.hf_dataset.select_columns(["episode_index"]).to_pandas().reset_index(drop=True)
    positions = {int(e): g.index.to_list() for e, g in pos.groupby("episode_index")}

    features = {}
    for k, v in src.meta.features.items():
        if k in DEFAULT_FEATURES:
            continue
        ft = {kk: vv for kk, vv in v.items() if kk != "info"}
        features[k] = ft
    log(f"features: {list(features)}")

    def make(name):
        root = out / "datasets" / name
        return LeRobotDataset.create(
            repo_id=f"{args.namespace}/{name}",
            fps=int(round(src.meta.fps)),  # source stores 10.0; the video encoder needs an integer rate
            features=features,
            root=root,
            robot_type=getattr(src.meta, "robot_type", None),
            use_videos=True,
            image_writer_threads=8,
            # LeRobot appends each episode's video onto one growing file and re-muxes the whole file
            # every time, so cost grows with file size. Small files keep every append cheap.
            # (Run 3 timed out at 4 h for this reason with the 500 MB default.)
            video_files_size_in_mb=10,
        )

    clean = make("libero_object_clean") if args.which in ("clean", "both") else None
    bad = make("libero_object_corrupted") if args.which in ("corrupted", "both") else None

    manifest = []
    t0 = time.time()
    for n, ep in enumerate(episodes):
        kind = assignment[ep]
        orig_task = obj_tasks[int(ep_task[ep])]
        task_bad = new_task_text.get(ep, orig_task)
        for row in positions[ep]:
            item = src[row]
            frame = {}
            for k in features:
                v = item[k]
                if k in (CAM_A, CAM_B):
                    # The reader returns channels-first (3, H, W). The writer validates against the
                    # feature shape, which is channels-last (H, W, 3), so transpose.
                    v = v.permute(1, 2, 0).contiguous()
                frame[k] = v.numpy() if hasattr(v, "numpy") else v
            frame_c = {**{k: np.copy(v) for k, v in frame.items()}, "task": orig_task}
            frame_b = {**{k: np.copy(v) for k, v in frame.items()}, "task": task_bad}
            if kind == "camera_swap":
                frame_b[CAM_A], frame_b[CAM_B] = frame[CAM_B].copy(), frame[CAM_A].copy()
            elif kind == "gripper_flip":
                frame_b["action"][GRIPPER_DIM] = -frame_b["action"][GRIPPER_DIM]
            if clean is not None:
                clean.add_frame(frame_c)
            if bad is not None:
                bad.add_frame(frame_b)
        if clean is not None:
            clean.save_episode()
        if bad is not None:
            bad.save_episode()
        manifest.append({
            "new_episode_index": n,
            "source_episode_index": ep,
            "source_task_index": int(ep_task[ep]),
            "corruption": kind,
            "original_task": orig_task,
            "corrupted_task": task_bad,
            "num_frames": len(positions[ep]),
        })
        if n % 10 == 0 or n == len(episodes) - 1:
            el = time.time() - t0
            log(f"[{args.which}] episode {n + 1}/{len(episodes)}  ({el / 60:.1f} min elapsed, ~{el / (n + 1) * (len(episodes) - n - 1) / 60:.1f} min left)")

    for ds in (clean, bad):
        if ds is not None:
            ds.finalize()
    log(f"finalized: {args.which}")
    if args.which == "clean":
        return  # the corrupted process writes the manifest (it's identical either way)

    with open(out / "corruption_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    json.dump({"source": SOURCE, "revision": revision, "seed": SEED, "rates": RATES, "counts": counts,
               "episodes": manifest}, open(out / "corruption_manifest.json", "w"), indent=1)
    log("wrote corruption_manifest.csv and .json")


if __name__ == "__main__":
    main()
