# Baseten Training Job: LIBERO + SmolVLA smoke test
# Checks that LeRobot, the LIBERO simulator, and SmolVLA training all run on Baseten
# before we spend money on the full clean-vs-corrupted experiment.
#
# Submit with:  baseten train push --config config.py
# Docs: https://docs.baseten.co/training/getting-started  and  https://docs.baseten.co/reference/sdk/training

from truss_train import definitions
from truss.base import truss_config

# Baseten's documented example image. run.sh installs Python 3.12 with uv,
# because current LeRobot needs Python 3.12 or newer.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        # A free Hugging Face read token, stored as a Baseten secret named hf_access_token.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # Headless rendering for the MuJoCo simulator. run.sh falls back to osmesa if EGL fails.
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    },
    # Keeps pip packages, the LIBERO dataset, and model weights between jobs.
    cache_config=definitions.CacheConfig(enabled=True),
    # Anything written to $BT_CHECKPOINT_DIR syncs to Baseten, so eval videos and logs survive the job.
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=1,
    cpu_count=16,      # the default is 1 CPU, too few for video decoding and the simulator
    memory="64Gi",     # the default is 2Gi
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
    ),
)

training_job = definitions.TrainingJob(
    name="libero-smolvla-smoke",
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name="voxel51-libero-smolvla",
    job=training_job,
)
