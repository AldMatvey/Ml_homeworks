import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import re


class AttentionVisualizer:
    def __init__(self, model, SRC, TRG):
        self.model = model
        self.SRC = SRC
        self.TRG = TRG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tokenize(self, text):
        tokens = [self.SRC.vocab.stoi.get(token, self.SRC.vocab.stoi['<unk>'])
                  for token in re.findall(r'\w+|[^\w\s]', text.lower())]
        return [self.SRC.vocab.stoi['<s>']] + tokens + [self.SRC.vocab.stoi['</s>']]

    def get_attention(self, src_tensor, trg_tensor):
        with torch.no_grad():
            src_emb = self.model.pos_encoder(self.model.src_embed(src_tensor) * math.sqrt(self.model.d_model))
            trg_emb = self.model.pos_encoder(self.model.trg_embed(trg_tensor) * math.sqrt(self.model.d_model))

            # Получаем выходы энкодера
            encoder_output = self.model.transformer.encoder(src_emb)

            # Проходим через каждый слой декодера
            attention_weights = []
            x = trg_emb
            for layer in self.model.transformer.decoder.layers:
                # Self-attention
                x = layer.self_attn(x, x, x, need_weights=False)[0]

                # Cross-attention (сохраняем веса)
                attn_output, attn_weights = layer.multihead_attn(
                    x, encoder_output, encoder_output,
                    need_weights=True
                )
                attention_weights.append(attn_weights)
                x = layer.norm1(x + attn_output)

                # Feed forward
                x = layer._ff_block(x)

            return attention_weights

    def visualize(self, text, max_len=30):
        # Токенизация
        src_tokens = self.tokenize(text)
        src_tensor = torch.tensor(src_tokens).unsqueeze(1).to(self.device)

        # Генерация с визуализацией внимания
        generated = [self.TRG.vocab.stoi['<s>']]
        src_text = [self.SRC.vocab.itos[i] for i in src_tokens]

        for step in range(max_len):
            trg_tensor = torch.tensor(generated).unsqueeze(1).to(self.device)

            # Получаем веса внимания
            attention_weights = self.get_attention(src_tensor, trg_tensor)
            last_layer_attention = attention_weights[-1][0]  # [1, num_heads, trg_len, src_len]

            # Генерация следующего слова
            with torch.no_grad():
                output = self.model(src_tensor, trg_tensor)
                next_word = output.argmax(-1)[-1].item()

            if next_word == self.TRG.vocab.stoi['</s>']:
                break

            generated.append(next_word)
            trg_text = [self.TRG.vocab.itos[i] for i in generated]

            # Визуализация для каждого head
            for head in range(last_layer_attention.size(1)):
                self.plot_attention(
                    last_layer_attention[0, head].cpu().numpy(),
                    src_text,
                    trg_text,
                    head=head,
                    step=step
                )

        return ' '.join([self.TRG.vocab.itos[i] for i in generated[1:]])

    def plot_attention(self, weights, src_tokens, trg_tokens, head=0, step=0):
        if weights.shape == ():
            return
        plt.figure(figsize=(10, 6))
        sns.heatmap(weights,
                    xticklabels=src_tokens,
                    yticklabels=trg_tokens,
                    cmap="YlGnBu",
                    linewidths=0.1,
                    linecolor='gray')
        plt.xlabel("Source Tokens")
        plt.ylabel("Target Tokens")
        plt.title(f"Step {step}, Head {head}")
        plt.tight_layout()
        plt.show()