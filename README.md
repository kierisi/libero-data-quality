# We broke LIBERO on purpose

What does messy training data cost a vision-language-action (VLA) model? We took the 454 LIBERO-Object demonstrations, planted three kinds of data problems in about 45% of them, fine-tuned SmolVLA on the clean and the corrupted copies with identical settings, and scored both in the LIBERO simulator.

| Training data | Success (200 simulated episodes) | 95% CI |
|---|---|---|
| Clean | **83.5%** (167/200) | 77.7 to 88.0 |
| Corrupted | **47.0%** (94/200) | 40.2 to 53.9 |

The gap is 36.5 points (two-proportion z = 7.67, p < 0.0001). Per-task results are in [`results/per_task.csv`](results/per_task.csv), and the full write-up is on the Voxel51 blog `[link added at publish]`.

## The datasets

Both are public LeRobot v3 datasets, Apache 2.0:

- [`kierisi/libero_object_clean`](https://huggingface.co/datasets/kierisi/libero_object_clean)
- [`kierisi/libero_object_corrupted`](https://huggingface.co/datasets/kierisi/libero_object_corrupted)

| Problem | Episodes | What changed |
|---|---|---|
| Instruction swap | 90 | The task text names a different LIBERO-Object item |
| Camera swap | 70 | The agent view and wrist view trade places |
| Gripper flip | 43 | Action dimension 6 (the gripper) is negated |
| Untouched | 251 | Nothing |

**How the corruptions were assigned:**
- Each problem went to separate episodes, stratified per task, with seed 51.
- Episode *i* is the same demonstration in both copies.
- Both copies went through the same decode and re-encode.

**The answer key** is in `answer_key/` in the corrupted repo, outside the dataset files. Try to find the problems yourself first.

## Reproduce it

Requirements:
- **Software:** Linux, Python 3.12, and the versions in [`requirements.txt`](requirements.txt).
- **GPU:** we trained on one 80 GB H100. Peak training memory was about 19 GB at batch 64, so smaller cards should work at a lower batch, but we only tested 80 GB.

```bash
pip install "lerobot[smolvla,libero]==0.6.1"
export MUJOCO_GL=egl          # use osmesa if your machine has no EGL
echo n | python -c "import libero.libero"   # answers LIBERO's first-run prompt

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

# Train (run once per copy: libero_object_clean and libero_object_corrupted)
lerobot-train --policy.path=lerobot/smolvla_base \
  --rename_map="$RENAME" --policy.empty_cameras=1 \
  --dataset.repo_id=kierisi/libero_object_clean --dataset.video_backend=torchcodec \
  --batch_size=64 --steps=10000 --save_freq=10000 --seed=1000 \
  --policy.push_to_hub=false --wandb.enable=false --policy.device=cuda \
  --output_dir=outputs/train_clean

# Evaluate in the LIBERO simulator: 20 episodes per task, 200 total
lerobot-eval --policy.path=outputs/train_clean/checkpoints/010000/pretrained_model \
  --rename_map="$RENAME" --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_object --eval.n_episodes=20 --eval.batch_size=20 \
  --env.init_states=true --seed=0 --policy.device=cuda --output_dir=outputs/eval_clean
```

**Run times:**
- Training: about 55 minutes per copy on one H100.
- Evaluation: 13 to 16 minutes with CPU rendering.

To rebuild the datasets from [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) instead of downloading them, see [`scripts/build_datasets.py`](scripts/build_datasets.py) and [`scripts/verify_datasets.py`](scripts/verify_datasets.py).

## Running it on Baseten

We ran everything as [Baseten Training Jobs](https://docs.baseten.co/training/overview). The job definitions are in [`baseten/`](baseten):

| Folder | What it is |
|---|---|
| `smoke` | The environment check |
| `pilot` | The clean baseline and step count |
| `build` | Builds and verifies both datasets |
| `experiment` | The main result |
| `ablation` | A ready-to-run follow-up that tests one problem at a time |

Submit any of them with `baseten train push --config config.py`. Any GPU provider works the same way. RunPod, Lambda, Modal, and Colab Pro+ are all options.

**Size the job below one H100's share.** One H100 comes with 16 CPU and about 118 GiB of memory. Our requests of 16 CPU / 64 GiB and above were scheduled on 2-GPU and 4-GPU instances, at 2 to 4 times the cost. The ablation config requests 8 CPU / 32 GiB, and it stops itself if it lands on more than one GPU.

## Gotchas we hit

1. **Frame layout:** LeRobot's dataset writer expects channels-last frames.
2. **Frame rate:** the video encoder needs an integer fps.
3. **Slow builds:** with the default 200 MB video files, every episode append rewrites a growing file, so the build slows down as it goes. We set `video_files_size_in_mb=10`.
4. **Rendering:** our training containers had no EGL, so the simulator rendered with OSMesa on the CPU.
5. **First-run prompt:** LIBERO asks an interactive question on first import. Answer it once, as shown above.
6. **Camera names:** `smolvla_base` expects cameras named `camera1` to `camera3`. LIBERO's are `image` and `image2`, so pass a rename map plus `--policy.empty_cameras=1`.
7. **Stale metadata:** LeRobot 0.6.1 sizes a dataset from `meta/info.json`. A stale count there made our corrupted run skip its last episode. That's 0.2% of the data, disclosed in [`results/main.json`](results/main.json), and fixed on the Hub.
8. **Pin from the real environment:** `pip freeze` inside a `uv venv` doesn't list the venv's packages. Pin versions from your install log instead.

## Credits and licenses

- **Data:**
  - LIBERO (Liu et al., 2023): the code is MIT and the demonstration data is CC BY 4.0.
  - [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) by the LeRobot team: Apache 2.0.
  - The derived datasets here: Apache 2.0.
- **Code:** this repository is Apache 2.0.
- **Compute:** training ran on Baseten.

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```
