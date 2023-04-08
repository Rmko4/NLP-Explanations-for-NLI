
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from esnli_data import ESNLIDataModule
from parse_args_T5_run import get_args
from t5_lit_classify import LitT5Classify
from t5_lit_module import LitT5
from model_output_processing import export_predictions

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
time = datetime.now().strftime("%m%d-%H:%M:%S")


def test(hparams):
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
    if hparams.results_save_path:
        hparams.results_save_path = os.path.expanduser(
            hparams.results_save_path)

    # Create data module
    data_module = ESNLIDataModule(
        model_name_or_path=hparams.model_name,
        dataset_path=hparams.data_path,
        eval_batch_size=hparams.eval_batch_size,
    )

    # Load model from checkpoint
    modelCls = LitT5 if not hparams.classify else LitT5Classify
    model = modelCls.load_from_checkpoint(
        checkpoint_path=hparams.checkpoint_load_path,
    )

    # Create trainer
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        limit_test_batches=hparams.limit_test_batches,
        limit_predict_batches=hparams.limit_predict_batches,
    )

    if hparams.predict:
        # Predict with model
        out = trainer.predict(model, datamodule=data_module)

        export_predictions(out, hparams, run_name)
    else:
        # Test model
        trainer.test(model, datamodule=data_module)


# Calling this script will do evaluation of the model on the test set
if __name__ == "__main__":
    hparams = get_args()
    hparams.predict = False
    test(hparams)
