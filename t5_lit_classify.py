# %%
import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import T5Tokenizer, T5EncoderModel
from torchmetrics import Accuracy
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self,
                 n_features: int = 512,
                 n_hidden: int = 256,
                 n_output: int = 3,
                 n_lstm_layers: int = 1,
                 lstm_droput: float = 0.5, ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=lstm_droput)
        self.classification_layer = nn.Sequential(nn.Linear(2*n_hidden, n_output),
                                                  torch.nn.Softmax(dim=-1))

    def forward(self, x):
        pass


class LitT5Classify(LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        tokenizer_path: str = "google/flan-t5-small",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        n_features: int = 512,
        n_hidden: int = 256,
        n_output: int = 3,
        n_lstm_layers: int = 1,
        lstm_droput: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        # self.model = None
        self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        self._freeze_encoder()

        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

        self.lstm, self.classification_layer = self._get_classification_head(n_features,
                                                                             n_hidden,
                                                                             n_output,
                                                                             n_lstm_layers,
                                                                             lstm_droput)
        self.loss = nn.CrossEntropyLoss()
        self.train_loss_history = []

        self.acc = Accuracy(task="multiclass", num_classes=n_output)

        # Does frame inspection so find init args
        self.save_hyperparameters()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _get_classification_head(self, n_features, n_hidden, n_output, layers, dropout):
        lstm = nn.LSTM(input_size=n_features,
                       hidden_size=n_hidden,
                       num_layers=layers,
                       batch_first=True,
                       bidirectional=True,
                       dropout=dropout)
        classification_layer = nn.Sequential(nn.Linear(2*n_hidden, n_output),
                                             torch.nn.Softmax(dim=-1))
        return lstm, classification_layer

    # def _get_model(model_name_or_path, self, n_features, n_hidden, n_output, layers, dropout):
    #     encoder = T5EncoderModel.from_pretrained(model_name_or_path)

    #     for param in encoder.parameters():
    #         param.requires_grad = False

    #     lstm = nn.LSTM(input_size=n_features,
    #                    hidden_size=n_hidden,
    #                    num_layers=layers,
    #                    batch_first=True,
    #                    bidirectional=True,
    #                    dropout=dropout)
    #     classification_layer = nn.Sequential(nn.Linear(2*n_hidden, n_output),
    #                                          torch.nn.Softmax(dim=-1))

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        # Pass input through the encoder
        outputs = self.encoder(input_ids)
        last_hidden_states = outputs.last_hidden_state
        # Pass the hidden states of the encoder through the lstm
        h_n, _ = self.lstm(last_hidden_states)
        # Pass the last hidden state of the lstm through the classification layer
        last_h_n = h_n[:, -1]
        out = self.classification_layer(last_h_n)
        return out

    def training_step(self, batch, batch_idx):
        y = batch['int_labels']
        out = self(batch)

        loss = self.loss(y, out)

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
        y = batch['int_labels']
        out = self(batch)

        val_loss = self.loss(y, out)

        y_hat = torch.argmax(out, dim=-1)
        self.acc.update(y, y_hat)

        self.log_dict({'val/loss': val_loss,
                       'val/acc': self.acc,
                       }, prog_bar=True)
        return {'val_loss': val_loss, }

    def configure_optimizers(self):
        # Might also add lr_scheduler
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


# model = LitT5Classify()
# print(model.parameters())
