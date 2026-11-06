"""Check that the corrupted copy differs from the clean copy exactly where the manifest says, and nowhere else."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAM_A, CAM_B = "observation.images.image", "observation.images.image2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--namespace", default="local")
    a = p.parse_args()
    out = Path(a.out)
    man = json.load(open(out / "corruption_manifest.json"))["episodes"]

    clean = LeRobotDataset(f"{a.namespace}/libero_object_clean", root=out / "datasets/libero_object_clean")
    bad = LeRobotDataset(f"{a.namespace}/libero_object_corrupted", root=out / "datasets/libero_object_corrupted")
    assert clean.meta.total_episodes == bad.meta.total_episodes == len(man), "episode counts differ"
    assert clean.meta.total_frames == bad.meta.total_frames, "frame counts differ"
    print(f"episodes: {len(man)}  frames: {clean.meta.total_frames}")

    def ep_rows(ds, ep):
        e = ds.meta.episodes[ep]
        return range(e["dataset_from_index"], e["dataset_to_index"])

    problems = 0
    checked = {"none": 0, "instruction_swap": 0, "camera_swap": 0, "gripper_flip": 0}
    for m in man:
        ep, kind = m["new_episode_index"], m["corruption"]
        rows = list(ep_rows(clean, ep))
        mid = rows[len(rows) // 2]
        c, b = clean[mid], bad[mid]
        ok = True
        if kind == "instruction_swap":
            ok = b["task"] == m["corrupted_task"] and c["task"] == m["original_task"] and b["task"] != c["task"]
        else:
            ok = b["task"] == c["task"]
        a_c, a_b = c["action"].numpy(), b["action"].numpy()
        if kind == "gripper_flip":
            ok &= np.allclose(a_b[6], -a_c[6]) and np.allclose(a_b[:6], a_c[:6])
        else:
            ok &= np.allclose(a_b, a_c)
        # Video re-encoding is lossy, so compare images by mean absolute difference.
        same = float((c[CAM_A] - b[CAM_A]).abs().mean())
        crossed = float((c[CAM_A] - b[CAM_B]).abs().mean())
        if kind == "camera_swap":
            ok &= crossed < same
        else:
            ok &= same < 0.05
        checked[kind] += 1
        if not ok:
            problems += 1
            print(f"PROBLEM episode {ep} ({kind}): task_c={c['task']!r} task_b={b['task']!r} same={same:.3f} crossed={crossed:.3f}")
    print(f"checked: {checked}")
    print("VERIFY: PASS" if problems == 0 else f"VERIFY: FAIL ({problems} episodes)")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
