from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


class LogGeneratedTextCallback(Callback):
    def __init__(self, n_samples: int = 20, log_every_n_steps=20) -> None:
        self.n_samples = n_samples
        self.processed_batches = 0
        self.log_every_n_steps = log_every_n_steps

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,):
        """Called when the validation batch ends."""
        self.processed_batches += 1
        # `outputs` comes from `LightningModule.validation_step`
        if self.processed_batches % self.log_every_n_steps == 0:
            # Create a dictionary with the required data
            data = {
                'input_text': outputs['input_text'][:self.n_samples],
                'generated_text': outputs['generated_text'][:self.n_samples],
                'reference_text_1': outputs['reference_texts'][0][:self.n_samples],
                'reference_text_2': outputs['reference_texts'][1][:self.n_samples],
                'reference_text_3': outputs['reference_texts'][2][:self.n_samples],
            }

            # Create a dataframe from the dictionary
            df = pd.DataFrame(data)

            # Log the texts to wandb
            logger: WandbLogger = trainer.logger
            logger.log_text('explanation_generation', dataframe=df)
