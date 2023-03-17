# -*- coding: utf-8 -*-
# @Time : 2023/3/16 14:28
# @Author : Jclian91
# @File : model.py
# @Place : Minghang, Shanghai
import math
import torch
import torch.nn as nn

from params import NUM_WORDS, EMBEDDING_SIZE


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# Text classifier based on a pytorch TransformerEncoder.
class TextClassifier(nn.Module):
    def __init__(
            self,
            nhead=8,
            dim_feedforward=2048,
            num_layers=6,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1
    ):
        super().__init__()

        vocab_size = NUM_WORDS + 2
        d_model = EMBEDDING_SIZE
        # vocab_size, d_model = embeddings.size()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # Embedding layer definition
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Linear(d_model, 5)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x
