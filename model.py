# -*- coding: utf-8 -*-
# @Time : 2023/9/11 20:05
# @Author: LZ
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        return x

class SENet(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SENet, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, length = x.size()
        pooled = self.pool(x).view(batch_size, channels, 1)
        weights = self.fc2(F.relu(self.fc1(pooled)))
        weights = torch.sigmoid(weights)
        weighted_x = x * weights.expand_as(x)
        return weighted_x

class SuperConvBlock(nn.Module):
    def __init__(self):

        super(SuperConvBlock, self).__init__()
        self.c1 = DilatedConvBlock(47, 16, kernel_size=3, dilation=1)
        self.c2 = DilatedConvBlock(47, 32, kernel_size=3, dilation=2)
        self.c3 = DilatedConvBlock(47, 64, kernel_size=3, dilation=4)
        self.senet1 = SENet(16)
        self.senet2 = SENet(32)
        self.senet3 = SENet(64)

    def forward(self, x):
        c1_out = self.c1(x)
        c2_out = self.c2(x)
        c3_out = self.c3(x)

        c1_out = self.senet1(c1_out)
        c2_out = self.senet2(c2_out)
        # print(c2_out.shape)
        c3_out = self.senet3(c3_out)
        # print(c3_out.shape)

        out = torch.cat([x, c1_out, c2_out, c3_out], dim=1)
        out = F.dropout(out, p=0.2)
        return out

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = DilatedConvBlock(159, 64, kernel_size=11, dilation=1)

    def forward(self, x):
        out = self.conv1(x)
        return out


class SERT_StructNet(nn.Module):
    def __init__(self, num_classes=3, num_heads=16):
        super(SERT_StructNet, self).__init__()
        self.super_conv_block = SuperConvBlock()
        self.conv_block = ConvBlock()
        self.gru = nn.GRU(64, 128, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        self.liner_gru = nn.Linear(256, 128)
        self.liner_lstm = nn.Linear(256, 128)

        self.liner_attention = nn.Linear(256, 128)

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=num_heads), num_layers=6)  # 指定Transformer编码器的层数

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_channels=47):
        x = input_channels.permute(0, 2, 1)
        x = self.super_conv_block(x)
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)

        gru_out, _ = self.gru(x)
        lstm_out, _ = self.lstm(x)
        gru_out = self.liner_gru(gru_out)
        lstm_out = self.liner_lstm(lstm_out)
        combined_out = torch.cat([gru_out, lstm_out], dim=2)
        x = self.dropout(combined_out)
        x = self.liner_attention(x)
        x = self.transformer_encoder(x)
        x = self.mlp(x)
        return x
