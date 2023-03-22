
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from esnli_data import ESNLIDataModule
from t5_lit_module import LitT5

# Make sure to login to wandb before running this script
# Run: wandb login

def main(hparams):
    # Create wandb logger
    wandb_logger = WandbLogger(
        name="Fine-Tuning",
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
    data_module = ESNLIDataModule(train_batch_size=8, eval_batch_size=64)

    # Create model
    model = LitT5()

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints/',
        # No f string formating yet
        filename='esnli-{epoch:02d}-{val/loss:.2f}',
    )

    callbacks = [checkpoint_callback]

    # Note that default behaviour does checkpointing for state of last training epoch
    callbacks = None

    # Create trainer
    trainer = Trainer(
        accelerator='auto',
        max_epochs=1,
        logger=wandb_logger,
        # Do validation every 50 steps
        val_check_interval=50,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main(hparams=None)
