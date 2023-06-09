
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from callbacks import LogGeneratedTextCallback
from esnli_data import ESNLIDataModule
from parse_args_T5_run import get_args
from t5_lit_module import LitT5
from t5_lit_classify import LitT5Classify

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
time = datetime.now().strftime("%m%d-%H:%M:%S")


def train(hparams):
    run_name = f"{hparams.run_name}_{time}"

    # Create wandb logger
    wandb_logger = WandbLogger(
        name=run_name,
        project="FLAN-T5-ESNLI",
        save_dir="logs/",
        log_model="all",
        anonymous="allow",
    )

    # Expands paths
    hparams.data_path = os.path.expanduser(hparams.data_path)
    if hparams.checkpoint_load_path:
        hparams.checkpoint_load_path = os.path.expanduser(
            hparams.checkpoint_load_path)
    if hparams.checkpoint_save_path:
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
    if not hparams.classify:
        if hparams.checkpoint_load_path:
            model = LitT5.load_from_checkpoint(
                checkpoint_path=hparams.checkpoint_load_path,
            )
        else:
            model = LitT5(model_name_or_path=hparams.model_name,
                          fine_tune_mode=hparams.fine_tune_mode,
                          learning_rate=hparams.learning_rate)
    else:
        model = LitT5Classify(model_name_or_path=hparams.model_name,
                              learning_rate=hparams.learning_rate,
                              checkpoint_path_main_model=hparams.checkpoint_load_path)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=hparams.checkpoint_save_path,
        filename=f'{run_name}-{time}-' + 'esnli-{epoch:02d}-{val/loss:.2f}',
        every_n_train_steps=hparams.val_check_interval,
    )

    callbacks = [checkpoint_callback]

    if not hparams.classify:
        # Add callback for logging text generation during training.
        log_generated_text_callback = LogGeneratedTextCallback(
            n_samples=hparams.n_text_samples,
            log_every_n_steps=hparams.log_every_n_generated)

        callbacks.append(log_generated_text_callback)

    # Create trainer
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=hparams.log_every_n_steps,
        val_check_interval=hparams.val_check_interval,
        limit_val_batches=hparams.limit_val_batches,
        callbacks=callbacks,
    )

    # Train Model
    # When training classifier we should not use checkpoint state for training of main model.
    ckpt_path = hparams.checkpoint_load_path if not hparams.classify else None
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
