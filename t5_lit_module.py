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
from typing import Union, List


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
        self.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path)
        self.generation_config.max_new_tokens = 128
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            model_name_or_path)

        self.blue_metric = textmetrics.BLEUScore()
        self.rouge_metric = textmetrics.ROUGEScore()
        self.perplexity_metric = textmetrics.Perplexity(ignore_index=-100)
        self.bert_metric = textmetrics.BERTScore()

        self.train_loss_history = []

        # Does frame inspection so find init args
        self.save_hyperparameters()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs: Seq2SeqLMOutput = self(**batch)
        loss = outputs.loss

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

        input_text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)
        generated_text = self.tokenizer.batch_decode(
            generated_out, skip_special_tokens=True)
        reference_texts = [self.tokenizer.batch_decode(
            batch[f'explanation_{i}'], skip_special_tokens=True) for i in range(1, 4)]

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

    def generate_text(self, input_text: Union[str, List[str]], max_length: int = 128, **generate_kwargs) -> Union[str, List[str]]:
        # Convert input_text to list if it is a string
        if isinstance(input_text, str):
            input_text = [input_text]

        # Encode input_text using the tokenizer
        input_ids = self.tokenizer.batch_encode_plus(
            input_text, return_tensors='pt', padding=True)['input_ids'].to(self.device)

        # Generate output texts using the model
        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            generation_config=self.generation_config,
            **generate_kwargs
        )

        # Decode the generated output texts using the tokenizer
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        # Return the generated output text as a list or string based on the input format
        if len(generated_text) == 1:
            return generated_text[0]
        else:
            return generated_text

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self._batch_generate(batch)
        return output

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output = self._batch_generate(batch)
        generated_text = output['generated_text']
        reference_texts = output['reference_texts']
        
        # Update suffices as we are only interested in epoch score
        self.blue_metric.update(generated_text, reference_texts)
        for i in range(3):
            self.bert_metric.update(generated_text, reference_texts[i])
            self.rouge_metric.update(generated_text, reference_texts[i])

        return output

    def test_epoch_end(self, outputs):
        # Log test blue score
        blue_score = self.blue_metric.compute()
        self.log('test/blue', blue_score)
        self.blue_metric.reset()

        # Log test bert score
        bert_score_dict = self.bert_metric.compute()
        bert_score = {k: sum(v) / len(v) for k, v in bert_score_dict.items()}
        self.log('test/bert', bert_score)
        self.bert_metric.reset()

        # Log test rouge score
        rouge_score = self.rouge_metric.compute()
        self.log('test/rouge', rouge_score)
        self.rouge_metric.reset()


    def _batch_generate(self, batch):
        generated_out = self.model.generate(
            batch['input_ids'], self.generation_config)

        generated_text = self.tokenizer.batch_decode(
            generated_out, skip_special_tokens=True)
        reference_texts = [self.tokenizer.batch_decode(
            batch[f'explanation_{i}'], skip_special_tokens=True) for i in range(1, 4)]
        input_text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)
        return {'input_text': input_text, 'generated_text': generated_text, 'reference_texts': reference_texts}


if __name__ == "__main__":
    model = LitT5('google/flan-t5-small')
