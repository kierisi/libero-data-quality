# Baseten job: ablation (one corruption at a time) for the LIBERO-Object experiment.
# Submit with:  baseten train push --config config.py
# Right after pushing, check the instance in the push output (or `baseten train job describe --job-id <id>`).
# run.sh also stops itself within a minute if it finds more than MAX_GPUS GPUs, so a wrong size costs cents.
# To run fewer arms (to save credits), edit ARMS below, e.g. "instruction_only".

from truss_train import definitions
from truss.base import truss_config

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        # Order matters only if the job is stopped early: results are written after each arm.
        "ARMS": "instruction_only,gripper_only,camera_only",
        "STEPS": "10000",
        "MAX_GPUS": "1",     # run.sh exits immediately if scheduled on a bigger instance (double the price)
        "KEEP_MODELS": "1",  # keep each arm's pretrained_model (~0.9 GB) so it can be re-evaluated without retraining
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_job = definitions.TrainingJob(
    name="libero-object-ablation",
    image=definitions.Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=definitions.Compute(
        node_count=1,
        # Baseten picks the instance from the CPU and memory request: 32 CPU / 128Gi -> H100:4 and
        # 16 CPU / 64Gi -> H100:2 in earlier jobs (an H100:2 reports 32 CPU / 236 GiB, so one H100 is
        # 16 CPU / 118 GiB and a request for all 16 does not fit). 8 CPU / 32Gi should land on a single H100.
        cpu_count=8,
        memory="32Gi",
        accelerator=truss_config.AcceleratorSpec(accelerator=truss_config.Accelerator.H100, count=1),
    ),
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name="voxel51-libero-smolvla", job=training_job)
