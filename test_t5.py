
import os
from datetime import datetime
from statistics import mean

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import CHRFScore
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text.rouge import ROUGEScore

from esnli_data import ESNLIDataModule
from parse_args_T5_run import get_args
from t5_lit_classify import LitT5Classify
from t5_lit_module import LitT5

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
time = datetime.now().strftime("%m%d-%H:%M:%S")


def evaluate(generated_texts, reference_texts):
    # implement scores
    chrf = CHRFScore()
    rouge = ROUGEScore()

    chrf_avg = 0
    rouge_avg = 0
    bert_avg = 0

    for target in reference_texts:
        rouge_avg += rouge(generated_texts, target)["rouge1_fmeasure"]
        chrf_avg += chrf(generated_texts, target)
        bert_avg += mean(bert_score(generated_texts, target)["f1"])

    rouge_avg /= len(reference_texts)
    chrf_avg /= len(reference_texts)
    bert_avg /= len(reference_texts)

    # implement F1 for CHRF and ROUGE
    f1 = 2 * (chrf_avg * rouge_avg) / (chrf_avg + rouge_avg)

    print(
        f"chrf={chrf_avg:.2f}, rouge={rouge_avg:.2f}, F1={f1:.2f}, bert={bert_avg:.2f}")


def main(hparams):
    run_name = f"{hparams.run_name}_{time}"

    # Create wandb logger
    wandb_logger = WandbLogger(
        name=run_name,
        project="FLAN-T5-ESNLI",
        save_dir="logs/",
        log_model="all"
    )

    hparams.data_path = os.path.expanduser(hparams.data_path)
    hparams.checkpoint_load_path = os.path.expanduser(
        hparams.checkpoint_load_path)
    hparams.results_save_path = os.path.expanduser(
        hparams.results_save_path)

    # Create data module
    data_module = ESNLIDataModule(
        model_name_or_path=hparams.model_name,
        dataset_path=hparams.data_path,
        eval_batch_size=hparams.eval_batch_size,
        classify=hparams.classify,
    )

    # Load model from checkpoint
    if not hparams.classify:
        model = LitT5.load_from_checkpoint(
            checkpoint_path=hparams.checkpoint_load_path,
        )
    else:
        model = LitT5Classify.load_from_checkpoint(
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

    # Test model
    # trainer.test(model, datamodule=data_module)

    # Predict with model
    out = trainer.predict(model, datamodule=data_module)

    if hparams.classify:
        # Remove batch dimension
        reference_labels = [batch['reference_label'] for batch in out]
        predicted_labels = [batch['predicted_label'] for batch in out]

        # Flatten lists
        reference_labels = [item for sublist in reference_labels for item in sublist]
        predicted_labels = [item for sublist in predicted_labels for item in sublist]

        # Make pandas dataframe out of reference and predicted label
        df = pd.DataFrame(
            list(zip(reference_labels, predicted_labels)),
            columns=['reference_label', 'predicted_label']
        )

        # Save as csv
        df.to_csv(f"{hparams.results_save_path}/{run_name}.csv", index=False)
    else:
        # Remove batch dimension
        input_texts = [batch['input_text'] for batch in out]
        generated_texts = [batch['generated_text'] for batch in out]
        reference_texts = [batch['reference_texts'] for batch in out]

        # Flatten lists
        input_texts = [item for sublist in input_texts for item in sublist]
        generated_texts = [item for sublist in generated_texts for item in sublist]

        # Flatten reference texts
        flattened_reference_texts = []
        for batch in reference_texts:
            reference_texts_t = list(map(list, zip(*batch)))
            flattened_reference_texts.extend(reference_texts_t)

        reference_texts = list(map(list, zip(*flattened_reference_texts)))

        print(input_texts)
        print(generated_texts)
        print(reference_texts)
        

        # Make pandas dataframe out of input, generated and reference texts
        df = pd.DataFrame(
            list(zip(input_texts, generated_texts, reference_texts[0], reference_texts[1], reference_texts[2])),
            columns=['input_text', 'generated_text', 'reference_texts_0', 'reference_texts_1', 'reference_texts_2']
        )

        # Save as csv
        df.to_csv(f"{hparams.results_save_path}/{run_name}.csv", index=False)



if __name__ == "__main__":
    hparams = get_args()
    main(hparams)
