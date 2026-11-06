# Baseten Training Job: LIBERO-Object pilot
# Fine-tunes smolvla_base on the clean LIBERO-Object episodes for 10k steps,
# then evaluates checkpoints in the LIBERO simulator (CPU rendering with OSMesa).
#
# Submit with:  baseten train push --config config.py

from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # The smoke test showed EGL isn't available in Baseten containers. OSMesa works.
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=1,
    cpu_count=32,      # OSMesa renders on CPU, so more cores = faster evaluation
    memory="128Gi",
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
    ),
)

training_job = definitions.TrainingJob(
    name="libero-object-pilot",
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name="voxel51-libero-smolvla",
    job=training_job,
)
