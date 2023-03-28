
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from callbacks import LogGeneratedTextCallback
from esnli_data import ESNLIDataModule
from parse_args_T5_run import get_args
from t5_lit_module import LitT5

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
time = datetime.now().strftime("%m%d-%H:%M:%S")
run_name = "Fine-Tuning_" + time


def main(hparams):
    # Create wandb logger
    wandb_logger = WandbLogger(
        name=run_name,
        project="FLAN-T5-ESNLI",
        save_dir="logs/",
        log_model="all",
    )

    hparams.data_path = os.path.expanduser(hparams.data_path)
    hparams.checkpoint_load_path = os.path.expanduser(
        hparams.checkpoint_load_path)
    hparams.checkpoint_save_path = os.path.expanduser(
        hparams.checkpoint_save_path)

    # Create data module
    data_module = ESNLIDataModule(
        model_name_or_path=hparams.model_name,
        dataset_path=hparams.data_path,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size,
    )

    # Create model
    if hparams.checkpoint_load_path:
        model = LitT5.load_from_checkpoint(
            checkpoint_path=hparams.checkpoint_load_path,
        )
    else:
        model = LitT5(model_name_or_path=hparams.model_name,
                      fine_tune_mode=hparams.fine_tune_mode,
                      learning_rate=hparams.learning_rate)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=hparams.checkpoint_save_path,
        # No f string formating yet
        filename=time + 'esnli-{epoch:02d}-{step}-{val/loss:.2f}',
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
    trainer.fit(model, data_module, ckpt_path=hparams.checkpoint_load_path)


if __name__ == "__main__":
    hparams = get_args()
    main(hparams)
