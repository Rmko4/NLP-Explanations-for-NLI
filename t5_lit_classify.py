import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, GenerationConfig, T5Tokenizer, T5Model, T5EncoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from torchmetrics.text.bert import BERTScore
# Import Bertscore, bleuscore and rougescore
import torchmetrics.text as textmetrics
from torch import nn


def dummy_metric(pred, gt):
    return {'eehh wadde': 1}


class LitT5Classify(LightningModule):
    def __init__(
        self,
        model_path: str = "google/flan-t5-small",
        tokenizer_path: str = "google/flan-t5-small",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        n_features: int = 512,
        n_hidden: int = 256,
        n_output: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.encoder = T5Model.from_pretrained(
            model_path)
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_path)
        self.classification_head = self.get_classification_head(n_features,
                                                                n_hidden,
                                                                n_output)

        self.loss = nn.CrossEntropyLoss()

        self.train_loss_history = []

        # Does frame inspection so find init args
        self.save_hyperparameters()

    def get_classification_head(self, n_features, n_hidden, n_output):
        return nn.Sequential(nn.Linear(n_features, n_hidden),
                             nn.ReLU(),
                             nn.Linear(n_hidden, n_output),
                             torch.nn.Softmax(dim=-1))

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        decoder_input_ids = self.encoder._shift_right(input_ids)
        outputs = self.encoder(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        last_hidden_states = outputs.last_hidden_state
        out = self.classification_head(last_hidden_states)
        return out

    def training_step(self, batch, batch_idx):
        y = batch['labels']
        out = self(**batch)
        y_hat = torch.argmax(out, dim=-1)

        loss = self.loss(y, y_hat)

        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('train/loss_step', loss, on_step=True,
                 on_epoch=False, prog_bar=True)

        self.train_loss_history.append(loss.item())

        if self.global_step % self.trainer.log_every_n_steps == 0 and self.global_step != 0:
            step_metrics = self.train_loss_history
            reduced = sum(step_metrics) / len(step_metrics)
            self.log('train/loss_step_reduced', reduced,
                     on_step=True, on_epoch=False, prog_bar=True)
            self.train_loss_history = []

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log('train/loss', loss, on_step=True,
        #          on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        generated_out = self.model.generate(
            batch['input_ids'], self.generation_config)

        generated_text = self.tokenizer.batch_decode(
            generated_out, skip_special_tokens=True)
        reference_texts = [self.tokenizer.batch_decode(
            batch[f'explanation_{i}'], skip_special_tokens=True) for i in range(1, 4)]
        input_text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)

        # Update suffices as we are only interested in epoch score
        self.blue_metric.update(generated_text, reference_texts)

        # for i in range(3):
        #     self.rouge_metric.update(
        #         generated_text, reference_texts[i])

        # This is only for validation on rightshifted explanation_1
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        val_loss = outputs.loss
        self.perplexity_metric.update(outputs.logits, batch['labels'])

        self.log_dict({'val/loss': val_loss,
                       'val/blue': self.blue_metric,
                       'val/perplexity': self.perplexity_metric,
                       }, prog_bar=True)
        # self.log_dict(self.rouge_metric, prog_bar=True)
        # self.log_dict(metric_dict, prog_bar=True)
        return {'val_loss': val_loss,
                'input_text': input_text,
                'generated_text': generated_text,
                'reference_texts': reference_texts}

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
