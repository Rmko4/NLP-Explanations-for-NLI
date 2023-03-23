import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, GenerationConfig, T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from torchmetrics.text.bert import BERTScore
# Import Bertscore, bleuscore and rougescore
import torchmetrics.text as textmetrics


def dummy_metric(pred, gt):
    return {'eehh wadde': 1}


class LitT5(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        tokenizer=None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path)
        self.metric = dummy_metric
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

        self.blue_metric = textmetrics.BLEUScore()
        self.bert_metric = textmetrics.BERTScore()

        # Does frame inspection so find init args
        self.save_hyperparameters()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs: Seq2SeqLMOutput = self(**batch)
        loss = outputs.loss

        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        # if self.trainer.val_check_interval % 50 == 0 and self.global_step != 0:
        #     step_metrics = self.logger.history['train/loss_step']
        #     reduced = sum(step_metrics) / len(step_metrics)
        #     self.logger.history['loss_step'] = []

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log('train/loss', loss, on_step=True,
        #          on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model.generate(batch['input_ids'], self.generation_config)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        reference_texts = [self.tokenizer.batch_decode(
            batch[f'explanation_{i}'], skip_special_tokens=True) for i in range(1, 4)]

        for i in range(3):
            self.blue_metric(generated_text, reference_texts[i])
            self.bert_metric(generated_text, reference_texts[i])

        # This is only for validation on rightshifted explanation_1
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        logits = outputs.logits
        val_loss = outputs.loss

        # Make sure to map pad token to -100

        # preds = torch.argmax(logits, dim=-1)

        # labels_1, labels_2, labels_3 = batch['labels_1'], batch['labels_2'], batch['labels_3']
        # metrics = [self.metric(preds, labels_1), self.metric(
        #     preds, labels_2), self.metric(preds, labels_3)]

        # # @NOTE ejj jooo minimizing, maximizing?
        # arg_max = np.argmax(
        #     [metrics[0]['eehh wadde'], metrics[1]['eehh wadde'], metrics[2]['eehh wadde']])
        # metric_dict = metrics[arg_max]

        self.log_dict({'val/loss': val_loss,
                       'val/blue': self.blue_metric,
                       'val/bert': self.bert_metric}, prog_bar=True)
        # self.log_dict(metric_dict, prog_bar=True)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        # Might also add lr_scheduler
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


if __name__ == "__main__":
    model = LitT5('google/flan-t5-small')
