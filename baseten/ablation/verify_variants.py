"""Check each single-corruption dataset differs from the clean copy exactly where the answer key says, and nowhere else.
Then push one batch through the same loading path lerobot-train uses, so a schema problem fails here (minutes)
instead of inside training (an hour of GPU time).

Expected content per variant:
  instruction_only: task = corrupted task, action = clean action, cameras = clean cameras
  gripper_only:     task = clean task, action = corrupted action, cameras = clean cameras
  camera_only:      task = clean task, action = clean action, cameras = corrupted cameras (swapped where listed)

Checks, per variant:
  1. every frame's task text and action vector (all 66,984 frames, read straight from the parquet columns)
  2. meta/stats.json action stats come from the right copy, state stats are unchanged
  3. the middle frame of every episode, decoded through LeRobotDataset (task string, action, agent camera)
  4. a DataLoader batch with SmolVLA's action chunk (50 steps), torchcodec, uint8 frames, spawn workers
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAM_A = "observation.images.image"
CHUNK = 50  # SmolVLA chunk_size: action delta indices 0..49


def load(name, root, **kw):
    ds = LeRobotDataset(f"local/{name}", root=root, **kw)
    return ds


def frame_tasks(ds):
    """Per-frame task text, the way the reader resolves it (tasks.parquet by position)."""
    ti = ds.hf_dataset.data.column("task_index").to_numpy()
    texts = np.array([str(s) for s in ds.meta.tasks.index])
    assert (ds.meta.tasks["task_index"].to_numpy() == np.arange(len(texts))).all()
    return texts[ti]


def frame_actions(ds):
    arr = ds.hf_dataset.data.column("action").combine_chunks()
    return arr.flatten().to_numpy().reshape(len(arr), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder holding clean/ and corrupted/")
    p.add_argument("--variants", required=True, help="folder holding libero_object_*/ and corruption_manifest.json")
    p.add_argument("--arms", default="instruction_only,gripper_only,camera_only")
    a = p.parse_args()
    src, var = Path(a.src), Path(a.variants)
    man = json.load(open(var / "corruption_manifest.json"))["episodes"]
    clean = load("clean", src / "clean")
    bad = load("corrupted", src / "corrupted")
    assert clean.meta.total_episodes == bad.meta.total_episodes == len(man), (clean.meta.total_episodes, bad.meta.total_episodes, len(man))
    assert clean.num_frames == bad.num_frames, (clean.num_frames, bad.num_frames)
    c_tasks, b_tasks = frame_tasks(clean), frame_tasks(bad)
    c_act, b_act = frame_actions(clean), frame_actions(bad)
    ep_of_frame = clean.hf_dataset.data.column("episode_index").to_numpy()
    kind = {m["new_episode_index"]: m["corruption"] for m in man}
    orig_task = {m["new_episode_index"]: m["original_task"] for m in man}
    bad_task = {m["new_episode_index"]: m["corrupted_task"] for m in man}

    # Sanity on the two sources themselves, against the answer key.
    exp_c = np.array([orig_task[e] for e in ep_of_frame])
    exp_b = np.array([bad_task[e] for e in ep_of_frame])
    assert (c_tasks == exp_c).all(), "clean copy task labels disagree with the answer key"
    assert (b_tasks == exp_b).all(), "corrupted copy task labels disagree with the answer key"
    flip_frames = np.array([kind[e] == "gripper_flip" for e in ep_of_frame])
    assert np.allclose(c_act[~flip_frames], b_act[~flip_frames]), "non-flipped actions differ between copies"
    assert np.allclose(c_act[flip_frames, :6], b_act[flip_frames, :6]) and np.allclose(c_act[flip_frames, 6], -b_act[flip_frames, 6]), \
        "gripper flips are not where the answer key says"
    print(f"sources OK: {len(man)} episodes, {clean.num_frames} frames, "
          f"{int(flip_frames.sum())} flipped frames, {int((c_tasks != b_tasks).sum())} relabeled frames", flush=True)

    expect = {
        "instruction_only": dict(task="bad", action="clean", cam="clean"),
        "gripper_only": dict(task="clean", action="bad", cam="clean"),
        "camera_only": dict(task="clean", action="clean", cam="bad"),
    }
    ref_tasks = {"clean": c_tasks, "bad": b_tasks}
    ref_act = {"clean": c_act, "bad": b_act}
    ref_ds = {"clean": clean, "bad": bad}
    failures = 0
    for name in a.arms.split(","):
        e = expect[name]
        root = var / f"libero_object_{name}"
        v = load(name, root)
        assert v.meta.total_episodes == len(man), (name, v.meta.total_episodes, len(man))
        assert v.num_frames == clean.num_frames, (name, v.num_frames, clean.num_frames)
        assert v.meta.features == clean.meta.features or v.meta.features == bad.meta.features

        # 1. whole-column checks
        t_ok = (frame_tasks(v) == ref_tasks[e["task"]]).all()
        a_ok = np.allclose(frame_actions(v), ref_act[e["action"]])
        # 2. stats used for normalization at train time
        s_ok = all(np.allclose(v.meta.stats["action"][k], ref_ds[e["action"]].meta.stats["action"][k]) for k in ("mean", "std", "min", "max"))
        s_ok &= all(np.allclose(v.meta.stats["observation.state"][k], clean.meta.stats["observation.state"][k]) for k in ("mean", "std"))
        if e["action"] == "bad":
            s_ok &= not np.allclose(v.meta.stats["action"]["mean"], clean.meta.stats["action"]["mean"])
        print(f"{name}: tasks {'OK' if t_ok else 'WRONG'}, actions {'OK' if a_ok else 'WRONG'}, stats {'OK' if s_ok else 'WRONG'}", flush=True)
        if not (t_ok and a_ok and s_ok):
            failures += 1

        # 3. decoded middle frame of every episode
        changed = {"task": 0, "action": 0, "cam": 0}
        bad_eps = 0
        for m in man:
            ep = m["new_episode_index"]
            info = clean.meta.episodes[ep]
            mid = (info["dataset_from_index"] + info["dataset_to_index"]) // 2
            c, b, x = clean[mid], bad[mid], v[mid]
            ref = {"clean": c, "bad": b}
            ok = x["task"] == ref[e["task"]]["task"]
            ok &= np.allclose(x["action"].numpy(), ref[e["action"]]["action"].numpy())
            ok &= float((x[CAM_A] - ref[e["cam"]][CAM_A]).abs().mean()) < 0.02
            changed["task"] += x["task"] != c["task"]
            changed["action"] += not np.allclose(x["action"].numpy(), c["action"].numpy())
            changed["cam"] += float((x[CAM_A] - c[CAM_A]).abs().mean()) > 0.05
            if not ok:
                bad_eps += 1
                if bad_eps <= 5:
                    print(f"PROBLEM {name} episode {ep} ({m['corruption']}): task={x['task']!r} expected {ref[e['task']]['task']!r}")
        counts = {k: sum(1 for m in man if m["corruption"] == k) for k in ("instruction_swap", "camera_swap", "gripper_flip")}
        exp_changed = {"task": counts["instruction_swap"] if e["task"] == "bad" else 0,
                       "action": counts["gripper_flip"] if e["action"] == "bad" else 0,
                       "cam": counts["camera_swap"] if e["cam"] == "bad" else 0}
        print(f"{name}: episodes differing from clean at mid frame -> {changed} (expected {exp_changed}), bad episodes: {bad_eps}", flush=True)
        if bad_eps or changed != exp_changed:
            failures += 1

        # 4. the training loading path: action chunk via delta_timestamps, torchcodec, uint8, spawn workers
        fps = v.meta.fps
        dl_ds = load(name, root, delta_timestamps={"action": [i / fps for i in range(CHUNK)]},
                     video_backend="torchcodec", return_uint8=True)
        rng = random.Random(0)
        idx = [rng.randrange(dl_ds.num_frames) for _ in range(16)]
        loader = torch.utils.data.DataLoader(dl_ds, batch_size=8, sampler=idx, num_workers=2,
                                             multiprocessing_context="spawn", persistent_workers=False)
        n_batches = 0
        for batch in loader:
            assert tuple(batch["action"].shape) == (8, CHUNK, 7), batch["action"].shape
            assert tuple(batch["action_is_pad"].shape) == (8, CHUNK)
            assert batch[CAM_A].dtype == torch.uint8 and tuple(batch[CAM_A].shape) == (8, 3, 256, 256), (batch[CAM_A].dtype, batch[CAM_A].shape)
            assert tuple(batch["observation.state"].shape) == (8, 8)
            assert isinstance(batch["task"], list) and all(isinstance(s, str) for s in batch["task"])
            n_batches += 1
        assert n_batches == 2
        print(f"{name}: dataloader smoke OK ({n_batches} batches, action chunk {CHUNK}, torchcodec, spawn)", flush=True)

    print("VERIFY: PASS" if failures == 0 else f"VERIFY: FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
