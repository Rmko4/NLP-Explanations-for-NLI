# Simple script to download an artifact (checkpoint) from wandb
import wandb

artifact_str = ...

run = wandb.init()
artifact = run.use_artifact(artifact_str, type='model')
artifact_dir = artifact.download()