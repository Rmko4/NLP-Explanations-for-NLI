from enum import Enum
from typing import List, Union

import numpy as np
import torch
# Import Bertscore, bleuscore and rougescore
import torchmetrics.text as textmetrics
import wandb
from peft import LoraConfig, TaskType, get_peft_model, PeftModelForSeq2SeqLM
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import (GenerationConfig, T5ForConditionalGeneration,
                          T5Tokenizer)
from transformers.modeling_outputs import Seq2SeqLMOutput


class FineTuneMode(str, Enum):
    FULL = 'full'
    LORA = 'lora'
    GRADUAL_UNFREEZING = 'gradual_unfreezing'


def get_ignore_params(fine_tune_mode: FineTuneMode):
    ignore_params = []
    if fine_tune_mode != FineTuneMode.LORA:
        ignore_params.extend(['lora_r', 'lora_alpha', 'lora_dropout'])
    return ignore_params


class LitT5(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        fine_tune_mode: Union[str, FineTuneMode] = FineTuneMode.FULL,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        # Does frame inspection so find init args
        self.save_hyperparameters(ignore=get_ignore_params(fine_tune_mode))

        self._load_model()
        self.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path)
        self.generation_config.max_new_tokens = 128
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            model_name_or_path)

        self.bleu_metric = textmetrics.BLEUScore(n_gram=3)
        self.chrf_metric = textmetrics.CHRFScore()
        self.rouge_metric = textmetrics.ROUGEScore()
        self.perplexity_metric = textmetrics.Perplexity(ignore_index=-100)
        self.bert_metric = textmetrics.BERTScore()

        self.train_loss_history = []

    def _load_model(self):
        hp = self.hparams
        self.model = T5ForConditionalGeneration.from_pretrained(
            hp.model_name_or_path)
        if hp.fine_tune_mode == FineTuneMode.LORA:
            self.peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=hp.lora_r,
                lora_alpha=hp.lora_alpha,
                lora_dropout=hp.lora_dropout)
            self.model: PeftModelForSeq2SeqLM = get_peft_model(
                self.model, self.peft_config)

    def get_encoder(self):
        if self.hparams.fine_tune_mode == FineTuneMode.LORA:
            return self.model.base_model.get_encoder()
        else:
            return self.model.get_encoder()

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
        output = self._batch_generate(batch)
        input_text = output['input_text']
        generated_text = output['generated_text']
        reference_texts = output['reference_texts']

        # Transpose the list of lists
        reference_texts_t = list(map(list, zip(*reference_texts)))

        # Update suffices as we are only interested in epoch score
        self.bleu_metric.update(generated_text, reference_texts_t)

        # This is only for validation on rightshifted explanation_1
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        val_loss = outputs.loss
        self.perplexity_metric.update(outputs.logits, batch['labels'])

        self.log_dict({'val/loss': val_loss,
                       'val/blue': self.bleu_metric,
                       'val/perplexity': self.perplexity_metric,
                       }, prog_bar=True)
        # self.log_dict(self.rouge_metric, prog_bar=True)
        # self.log_dict(metric_dict, prog_bar=True)
        return {'val_loss': val_loss,
                'input_text': input_text,
                'generated_text': generated_text,
                'reference_texts': reference_texts_t}

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

        # Transpose the list of lists
        reference_texts_t = list(map(list, zip(*reference_texts)))

        # Update suffices as we are only interested in epoch score
        self.bleu_metric.update(generated_text, reference_texts_t)
        self.chrf_metric.update(generated_text, reference_texts_t)
        for i in range(3):
            self.bert_metric.update(generated_text, reference_texts[i])
            self.rouge_metric.update(generated_text, reference_texts[i])

        return output

    def on_test_epoch_end(self):
        # Log test blue score
        blue_score = self.bleu_metric.compute()
        self.log('test/blue', blue_score)
        self.bleu_metric.reset()

        # Log test chrf score
        chrf_score = self.chrf_metric.compute()
        self.log('test/chrf', chrf_score)
        self.chrf_metric.reset()

        # Log test bert score
        bert_score_dict = self.bert_metric.compute()
        bert_score = {f'test/bert/{k}': sum(v) / len(v) for k, v in bert_score_dict.items()}
        self.log_dict(bert_score)
        self.bert_metric.reset()

        # Log test rouge score
        rouge_score = self.rouge_metric.compute()
        # Add test/rouge prefix to the keys
        rouge_score = {f'test/rouge/{k}': v for k, v in rouge_score.items()}
        self.log_dict(rouge_score)
        self.rouge_metric.reset()

    def _batch_generate(self, batch):
        generated_out = self.model.generate(
            inputs=batch['input_ids'], generation_config=self.generation_config)

        generated_text = self.tokenizer.batch_decode(
            generated_out, skip_special_tokens=True)
        reference_texts = [self.tokenizer.batch_decode(
            batch[f'explanation_{i}'], skip_special_tokens=True) for i in range(1, 4)]
        input_text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)
        return {'input_text': input_text, 'generated_text': generated_text, 'reference_texts': reference_texts}


if __name__ == "__main__":
    model = LitT5('google/flan-t5-small')
