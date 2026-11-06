# Baseten job: clean vs corrupted LIBERO-Object (the main experiment).
# Run only after the build job has pushed kierisi/libero_object_clean and kierisi/libero_object_corrupted.
# Submit with:  baseten train push --config config.py

from truss_train import definitions
from truss.base import truss_config

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "DATA_NAMESPACE": "kierisi",
        "ARMS": "clean,corrupted",   # run both arms in sequence
        "STEPS": "10000",            # pilot: 5k = 89%, 10k = 91%, so 10k is enough
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_job = definitions.TrainingJob(
    name="libero-object-clean-vs-corrupted",
    image=definitions.Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=definitions.Compute(
        node_count=1,
        cpu_count=16,      # sized to stay on a single-H100 instance (32 CPU / 128Gi was scheduled as 4x H100)
        memory="64Gi",
        accelerator=truss_config.AcceleratorSpec(accelerator=truss_config.Accelerator.H100, count=1),
    ),
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name="voxel51-libero-smolvla", job=training_job)
