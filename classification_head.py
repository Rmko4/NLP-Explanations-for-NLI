from torch import nn
import torch


class ClassificationHeadRNN(nn.Module):
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
        h_n, _ = self.lstm(x)
        # Pass the last hidden state of the lstm through the classification layer
        last_h_n = h_n[:, -1]
        out = self.classification_layer(last_h_n)
        return out


class ClassificationHeadAttn(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 n_hidden: int = 256,
                 n_output: int = 3,
                 m_h_att_dropout: float = 0.2,):
        super().__init__()
        self.m_h_attn = nn.MultiheadAttention(embed_dim,
                                              num_heads=1,
                                              dropout=m_h_att_dropout,
                                              batch_first=True)
        self.l1 = nn.Linear(embed_dim, n_hidden)
        self.ReLU = nn.ReLU()
        self.l_out = nn.Linear(n_hidden, n_output)

    def _preprocess_attn_mask(self, attn_mask):
        x = ~attn_mask.bool()
        x = torch.unsqueeze(x, dim=-1)
        x = x.repeat(1, 1, attn_mask.shape[-1])
        return x

    def forward(self, x, attn_mask):
        bool_attn_mask = self._preprocess_attn_mask(attn_mask)
        attn_out, _ = self.m_h_attn(x, x, x, attn_mask=bool_attn_mask)
        # average over the sequence dimension
        avg_pool = torch.nanmean(attn_out, dim=1)
        x = self.l1(avg_pool)
        x = self.ReLU(x)
        logits = self.l_out(x)
        return logits


if __name__ == '__main__':
    from esnli_data import ESNLIDataModule
    import os
    dataset_path = os.path.expanduser('~/datasets/esnli_classify')

    dm = ESNLIDataModule(classify=True, dataset_path=dataset_path)
    dm.setup()

    # Prints the first batch of the training set
    data = next(iter(dm.train_dataloader()))
    attn_mask = data['attention_mask']

    head = ClassificationHeadAttn()
    x = torch.randn(16, attn_mask.shape[1], 512)
    logits = head(x, attn_mask)
    pass
