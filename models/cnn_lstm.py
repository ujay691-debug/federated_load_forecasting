import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        total_pad = self.kernel_size - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="constant", value=float("-inf"))
        return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)


class Attention(nn.Module):
    def __init__(self, input_dim: int, attn_units: int):
        super().__init__()
        self.score_vec = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_out = nn.Linear(input_dim * 2, attn_units, bias=False)

    def forward(self, x):
        score_first_part = self.score_vec(x)
        h_t = x[:, -1, :]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        pre_activation = torch.cat([context, h_t], dim=1)
        attn_vector = torch.tanh(self.attn_out(pre_activation))
        return attn_vector


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_attention = cfg.use_attention

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cfg.conv1_channels,
            kernel_size=cfg.conv1_kernel,
            stride=1,
            padding=cfg.conv1_kernel // 2,
        )
        self.pool1 = SamePadMaxPool1d(kernel_size=cfg.pool1_kernel, stride=1)

        self.conv2 = nn.Conv1d(
            in_channels=cfg.conv1_channels,
            out_channels=cfg.conv2_channels,
            kernel_size=cfg.conv2_kernel,
            stride=1,
            padding=cfg.conv2_kernel // 2,
        )
        self.pool2 = SamePadMaxPool1d(kernel_size=cfg.pool2_kernel, stride=1)

        self.dropout = nn.Dropout(cfg.dropout)

        self.lstm1 = nn.LSTM(
            input_size=cfg.conv2_channels,
            hidden_size=cfg.lstm_hidden1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=cfg.lstm_hidden1,
            hidden_size=cfg.lstm_hidden2,
            batch_first=True,
        )

        if self.use_attention:
            self.attention = Attention(input_dim=cfg.lstm_hidden2, attn_units=cfg.attn_units)
            self.fc1 = nn.Linear(cfg.attn_units, cfg.fc_hidden)
        else:
            self.fc1 = nn.Linear(cfg.lstm_hidden2, cfg.fc_hidden)

        self.fc2 = nn.Linear(cfg.fc_hidden, output_dim)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        if self.use_attention:
            x = self.attention(x)
        else:
            x = x[:, -1, :]

        return x

    def forward_head(self, feat):
        out = F.relu(self.fc1(feat))
        out = self.fc2(out)
        return out

    def forward(self, x, return_feature=False):
        feat = self.extract_features(x)
        pred = self.forward_head(feat)
        if return_feature:
            return pred, feat
        return pred
