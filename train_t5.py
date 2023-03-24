
from datetime import datetime
import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from esnli_data import ESNLIDataModule
from t5_lit_module import LitT5
from callbacks import LogGeneratedTextCallback
from parse_args_train import get_args


# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
run_name = "Fine-Tuning_" + datetime.now().strftime("%m%d-%H:%M:%S")
# data_path = "~/datasets/esnli/"
# data_path = os.path.expanduser(data_path)
# model_name = "google/flan-t5-base"


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
        model_name_or_path=hparams.model_name,
        dataset_path=hparams.data_path,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size,
    )

    # Create model
    model = LitT5(model_name_or_path=hparams.model_name,
                  learning_rate=hparams.learning_rate)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=hparams.checkpoint_path,
        # No f string formating yet
        filename='esnli-{epoch:02d}-{val/loss:.2f}',
        every_n_train_steps=hparams.val_check_interval,
    )
    log_generated_text_callback = LogGeneratedTextCallback(
        n_samples=hparams.n_text_samples,
        log_every_n_steps=hparams.log_every_n_generated)

    callbacks = [checkpoint_callback, log_generated_text_callback]

    # Note that default behaviour does checkpointing for state of last training epoch
    # callbacks = None

    # Create trainer
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=hparams.log_every_n_steps,
        # Do validation every 50 steps
        val_check_interval=hparams.val_check_interval,
        limit_val_batches=hparams.limit_val_batches,
        callbacks=callbacks,
    )

    # Validate
    # trainer.validate(model, data_module)

    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    hparams = get_args()
    main(hparams)
