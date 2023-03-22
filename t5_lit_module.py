import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.optim import AdamW


def dummy_metric(pred, gt):
    return {'eehh wadde': 1}


class LitT5(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path)
        self.metric = dummy_metric

        # Does frame inspection so find init args
        self.save_hyperparameters()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs: Seq2SeqLMOutput = self(**batch)
        loss = outputs.loss
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train/loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # This is only for validation on rightshifted explanation_1
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        
        logits = outputs.logits
        val_loss = outputs.loss


        # preds = torch.argmax(logits, axis=1)

        # labels_1, labels_2, labels_3 = batch['labels_1'], batch['labels_2'], batch['labels_3']
        # metrics = [self.metric(preds, labels_1), self.metric(
        #     preds, labels_2), self.metric(preds, labels_3)]

        # # @NOTE ejj jooo minimizing, maximizing?
        # arg_max = np.argmax(
        #     [metrics[0]['eehh wadde'], metrics[1]['eehh wadde'], metrics[2]['eehh wadde']])
        # metric_dict = metrics[arg_max]

        self.log('val/loss', val_loss, prog_bar=True)
        # self.log_dict(metric_dict, prog_bar=True)

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
