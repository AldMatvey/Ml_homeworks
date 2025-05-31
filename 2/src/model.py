import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import gensim

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, padding_idx, smoothing=0.1, dim=-1):  # Изменили 'classes' → 'num_classes'
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes  # Исправлено имя
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 2))  # Используем новое имя
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            mask = (target.data == self.padding_idx)
            true_dist[mask] = 0.0
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TransformerModel(nn.Module):
    def __init__(self, vocab_sizes, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, pretrained_embeddings_path=None, SRC = None, TRG = None):
        super().__init__()
        self.save_attention = False  # По умолчанию отключено
        self.attention_weights = None
        self.d_model = d_model
        self.nhead = nhead
        self.src_pad_idx = None
        self.trg_pad_idx = None

        # 1. Явная инициализация трансформера
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,  # Обязательный параметр
            dropout=0.1
        )

        # 2. Эмбеддинги и позиционное кодирование
        self.src_embed = nn.Embedding(vocab_sizes['src'], d_model)
        self.trg_embed = nn.Embedding(vocab_sizes['trg'], d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Выходной слой
        self.fc_out = nn.Linear(d_model, vocab_sizes['trg'])

        # Инициализация эмбеддингов с предобученными весами
        if pretrained_embeddings_path != None:
            self.src_embed = self._init_embedding(
                vocab_size=vocab_sizes['src'],
                d_model=d_model,
                vocab=SRC.vocab,  # Добавить передачу словаря
                emb_path=pretrained_embeddings_path
            )

            self.trg_embed = self._init_embedding(
                vocab_size=vocab_sizes['trg'],
                d_model=d_model,
                vocab=TRG.vocab,
                emb_path=pretrained_embeddings_path
            )

    def _init_embedding(self, vocab_size, d_model, vocab, emb_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        embed = nn.Embedding(vocab_size, d_model).to(device)

        if emb_path:
            # Загрузка предобученных эмбеддингов (пример для FastText)
            pretrained = gensim.models.fasttext.load_facebook_model(emb_path)

            # Инициализация весов
            weights = torch.randn(vocab_size, d_model).to(device)
            found = 0

            for word, idx in vocab.stoi.items():
                if word in pretrained.wv:
                    weights[idx] = torch.from_numpy(pretrained.wv[word]).to(device)
                    found += 1

            print(f"Loaded {found}/{vocab_size} vectors")
            embed.weight.data.copy_(weights)
            embed.weight.requires_grad = True  # Разрешаем дообучение

        return embed

        # Трансформер
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.3
        )

        self.fc_out = nn.Linear(d_model, vocab_sizes['trg'])
        self.save_attention = False
        self.attention_weights = None

    def generate_square_subsequent_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf')).transpose(0, 1)

    def forward(self, src, trg):
        # src: [S, B], trg: [T, B]
        assert src.dim() == 2, f"Expected src shape [S, B], got {src.shape}"
        assert trg.dim() == 2, f"Expected trg shape [T, B], got {trg.shape}"
        print(f"Input shapes - src: {src.shape}, trg: {trg.shape}")
        src_emb = self.src_embed(src)
        print(f"After embedding - src: {src_emb.shape}")
        S, B = src.size()
        T = trg.size(0)

        # Маска для исходной последовательности (паттерны)
        src_key_padding_mask = (src == self.src_pad_idx).transpose(0, 1)  # [B, S]

        # Маска для целевой последовательности
        tgt_key_padding_mask = (trg == self.trg_pad_idx).transpose(0, 1)  # [B, T]
        tgt_mask = self.generate_square_subsequent_mask(T, trg.device)  # [T, T]

        # Эмбеддинг + позиционное кодирование
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.trg_embed(trg) * math.sqrt(self.d_model))

        # Forward pass
        output = self.transformer(
            src_emb,
            trg_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        # Модифицируем вызов трансформера для сохранения весов внимания
        output = self.transformer(
            src_emb,
            trg_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Сохраняем веса внимания, если нужно
        if self.save_attention:
            self.attention_weights = self.transformer.decoder.layers[-1].multihead_attn.attn_weights

        return self.fc_out(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)