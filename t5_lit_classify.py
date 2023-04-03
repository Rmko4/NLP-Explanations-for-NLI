import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy
from transformers import T5EncoderModel, T5Tokenizer

from t5_lit_module import LitT5
from classification_head import ClassificationHeadAttn


class LitT5Classify(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        checkpoint_path: str = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        n_hidden: int = 256,
        n_output: int = 3,
        m_h_attn_dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        if checkpoint_path:
            model = LitT5.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
            )
            self.encoder = model.get_encoder()
        else:
            self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        self._freeze_encoder()

        embed_dim = self.encoder.config.d_model
        self.classification_head = ClassificationHeadAttn(embed_dim,
                                                          n_hidden,
                                                          n_output,
                                                          m_h_attn_dropout)

        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

        self.loss = nn.CrossEntropyLoss()
        self.train_loss_history = []

        self.acc_metric = Accuracy(task="multiclass", num_classes=n_output)

        # Does frame inspection so find init args
        self.save_hyperparameters()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # Pass input through the encoder
        outputs = self.encoder(input_ids)
        last_hidden_states = outputs.last_hidden_state
        # Pass the hidden states through the classification head
        out = self.classification_head(last_hidden_states, attention_mask)
        return out

    def training_step(self, batch, batch_idx):
        y_target = batch['int_labels']
        logits = self(batch)

        loss = self.loss(logits, y_target)

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

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y_target = batch['int_labels']
        logits = self(batch)

        val_loss = self.loss(logits, y_target)

        y_hat = torch.argmax(logits, dim=-1)
        self.acc_metric.update(y_hat, y_target)

        self.log_dict({'val/loss': val_loss,
                       'val/acc': self.acc_metric,
                       }, prog_bar=True)
        return {'val_loss': val_loss,
                'y_hat': y_hat}

    def configure_optimizers(self):
        # Might also add lr_scheduler
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        optimizer = AdamW(
            self.classification_head.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
