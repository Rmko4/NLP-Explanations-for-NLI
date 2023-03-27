
from datetime import datetime
import os

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
time = datetime.now().strftime("%m%d-%H:%M:%S")
run_name = "Testing_" + time


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

    hparams.data_path = os.path.expanduser(hparams.data_path)
    hparams.checkpoint_path = os.path.expanduser(hparams.checkpoint_path)

    # Create data module
    data_module = ESNLIDataModule(
        model_name_or_path=hparams.model_name,
        dataset_path=hparams.data_path,
        eval_batch_size=hparams.eval_batch_size,
    )

    # Load model from checkpoint
    model = LitT5.load_from_checkpoint(
        checkpoint_path=hparams.checkpoint_path,
    )

    # Create trainer
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        limit_test_batches=hparams.limit_test_batches,
        limit_predict_batches=hparams.limit_predict_batches,
    )

    # Test model
    trainer.test(model, datamodule=data_module)

    # Predict with model
    # out = trainer.predict(model, datamodule=data_module)
    # input_texts = out[0]['input_text']
    # generated_texts = out[0]['generated_text']
    # reference_texts = out[0]['reference_texts']

if __name__ == "__main__":
    hparams = get_args()
    main(hparams)
