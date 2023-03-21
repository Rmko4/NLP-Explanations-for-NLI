from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration
import torch
import numpy as np


def dummy_metric(pred, gt):
    return {'eehh wadde': 1}


class LitT5(LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.metric = dummy_metric

    def forward(self, **inputs):
        return self.model(**inputs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        preds = torch.argmax(logits, axis=1)

        labels_1, labels_2, labels_3 = batch['labels_1'], batch['labels_2'], batch['labels_3']
        metrics = [self.metric(preds, labels_1), self.metric(preds, labels_2), self.metric(preds, labels_3)]

        # @NOTE ejj jooo minimizing, maximizing?
        arg_max = np.argmax([metrics[0]['eehh wadde'], metrics[1]['eehh wadde'], metrics[2]['eehh wadde']])
        metric_dict = metrics[arg_max]

        self.log('val_loss', val_loss, prog_bar=True)
        self.log_dict(metric_dict, prog_bar=True)


if __name__ == '__main__':
    model = LitT5('t5-small')
