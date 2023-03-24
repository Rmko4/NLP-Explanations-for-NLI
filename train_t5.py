
from datetime import datetime

import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from esnli_data import ESNLIDataModule
from t5_lit_module import LitT5
from callbacks import LogGeneratedTextCallback

import os

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
run_name = "Fine-Tuning_" + datetime.now().strftime("%m%d-%H:%M:%S")
data_path = "~/datasets/esnli/"
data_path = os.path.expanduser(data_path)


def main(hparams):
    # Create wandb logger
    wandb_logger = WandbLogger(
        name=run_name,
        project="FLAN-T5-ESNLI",
        save_dir="logs/",
        log_model="all"
    )

    # To log additional params, outside lightning module hparams:
    # add one parameter
    # wandb_logger.experiment.config["key"] = value
    # add multiple parameters
    # wandb_logger.experiment.config.update({key1: val1, key2: val2})

    # Create data module
    data_module = ESNLIDataModule(
        train_batch_size=64, eval_batch_size=4, dataset_path=data_path)

    # Create model
    model = LitT5()

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints/',
        # No f string formating yet
        filename='esnli-{epoch:02d}-{val/loss:.2f}',
        every_n_train_steps=1000,
    )
    log_generated_text_callback = LogGeneratedTextCallback(n_samples=10)

    callbacks = [checkpoint_callback, log_generated_text_callback]

    # Note that default behaviour does checkpointing for state of last training epoch
    # callbacks = None

    # Create trainer
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=3,
        logger=wandb_logger,
        log_every_n_steps=50,
        # Do validation every 50 steps
        val_check_interval=200,
        limit_val_batches=20,
        callbacks=callbacks,
    )

    # Validate
    trainer.validate(model, data_module)

    # Train
    # trainer.fit(model, data_module)


if __name__ == "__main__":
    main(hparams=None)
