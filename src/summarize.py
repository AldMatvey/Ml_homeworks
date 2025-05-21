import re
import torch
from torch.nn.functional import log_softmax

import spacy
from spacy.lang.ru.stop_words import STOP_WORDS
from heapq import nlargest


def generate_summary(model, SRC, TRG, text, max_len=30, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()

    # Токенизация
    tokens = [SRC.vocab.stoi.get(token, SRC.vocab.stoi['<unk>'])
              for token in re.findall(r'\w+|[^\w\s]', text.lower())]
    tokens = [SRC.vocab.stoi['<s>']] + tokens + [SRC.vocab.stoi['</s>']]

    # Подготовка входных данных (src)
    src = torch.tensor(tokens).unsqueeze(1).to(device)  # [S, 1]

    # Генерация заголовка
    generated = [TRG.vocab.stoi['<s>']]
    print("\nGeneration debug:")
    print("Input tokens:", [SRC.vocab.itos[i] for i in tokens])
    print("First decoder output:", generated[0].softmax(-1).topk(5))

    for _ in range(max_len):
        trg = torch.tensor(generated).unsqueeze(1).to(device)  # [T, 1]

        with torch.no_grad():
            output = model(src, trg)  # Используем полную модель, как в forward()

        next_word = output.argmax(dim=-1)[-1].item()
        if next_word == TRG.vocab.stoi['</s>']:
            break

        generated.append(next_word)

    # Преобразование в текст
    return ' '.join(TRG.vocab.itos[i] for i in generated[1:] if i not in [
        TRG.vocab.stoi['<s>'], TRG.vocab.stoi['</s>'], TRG.vocab.stoi['<pad>']
    ])


def beam_search(model, src, SRC, TRG, beam_width=3, max_len=30, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    src = src.to(device)

    # Энкодинг исходного текста
    with torch.no_grad():
        src_emb = model.pos_encoder(model.src_embed(src))
        encoder_output = model.transformer.encoder(src_emb)

    # Инициализация beam-поиска
    beams = [([TRG.vocab.stoi['<s>']], 0)]  # (последовательность, суммарный log-шанс)

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == TRG.vocab.stoi['</s>']:
                new_beams.append((seq, score))
                continue

            trg = torch.tensor(seq).unsqueeze(1).to(device)
            with torch.no_grad():
                trg_emb = model.pos_encoder(model.trg_embed(trg))
                output = model.transformer.decoder(trg_emb, encoder_output)
                logits = model.fc_out(output[-1])
                log_probs = log_softmax(logits, dim=-1)
                topk = torch.topk(log_probs, beam_width, dim=-1)

            for i in range(beam_width):
                new_seq = seq + [topk.indices[0][i].item()]
                new_score = score + topk.values[0][i].item()
                new_beams.append((new_seq, new_score))

        # Выбор топ-N гипотез
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beams[0][0]
    return ' '.join([TRG.vocab.itos[i] for i in best_seq[1:-1]])


def greedy_decoding(model, SRC, TRG, text, max_len=30, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    tokens = [SRC.vocab.stoi.get(token, SRC.vocab.stoi['<unk>'])
              for token in re.findall(r'\w+|[^\w\s]', text.lower())]
    tokens = [SRC.vocab.stoi['<s>']] + tokens + [SRC.vocab.stoi['</s>']]
    src = torch.tensor(tokens).unsqueeze(1).to(device)

    generated = [TRG.vocab.stoi['<s>']]
    for _ in range(max_len):
        trg = torch.tensor(generated).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(src, trg)
            next_word = output.argmax(-1)[-1].item()

        if next_word == TRG.vocab.stoi['</s>']:
            break
        generated.append(next_word)

    return ' '.join(TRG.vocab.itos[i] for i in generated[1:])


nlp = spacy.load("ru_core_news_lg")  # Для русского языка


def spacy_summary(text, num_sentences=1):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    word_frequencies = {}

    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.isalpha():
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1

    max_freq = max(word_frequencies.values()) if word_frequencies else 1
    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text]

    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return " ".join([sent.text for sent in summary_sentences])