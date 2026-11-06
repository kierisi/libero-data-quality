# Baseten job: build the clean and deliberately corrupted LIBERO-Object datasets.
# Submit with:  baseten train push --config config.py
#
# To also push both datasets to the Hugging Face Hub (private), set PUSH_NAMESPACE below to your
# HF username or org, and make sure the hf_access_token secret is a WRITE token.

from truss_train import definitions
from truss.base import truss_config

PUSH_NAMESPACE = "kierisi"  # your Hugging Face username. "local" = don't push, keep in job outputs only.
PUSH_PRIVATE = "false"     # "false" = public, so tutorial readers can follow along. "true" = private.

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "PUSH_NAMESPACE": PUSH_NAMESPACE,
        "PUSH_PRIVATE": PUSH_PRIVATE,
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_job = definitions.TrainingJob(
    name="libero-object-build-datasets",
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
