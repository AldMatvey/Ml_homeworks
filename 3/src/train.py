from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchtext.data import Field, Example, Dataset, BucketIterator
from src.summarize import generate_summary
from rouge_score import rouge_scorer
from summarize import greedy_decoding
from model import LabelSmoothingLoss

def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }


def evaluate(model, data_loader, criterion, device, SRC, TRG):
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for src, trg in data_loader:
            # Вычисление loss
            output = model(src, trg[:-1])
            loss = criterion(output.view(-1, output.shape[-1]),
                             trg[1:].view(-1))
            total_loss += loss.item()

            # Генерация предсказаний для ROUGE
            batch_predictions = []
            for i in range(src.size(1)):
                src_seq = src[:, i]
                trg_seq = trg[:, i]

                # Генерация предсказания
                pred_tokens = greedy_decoding(model, src_seq.unsqueeze(1), SRC, TRG, device)
                pred_text = ' '.join(pred_tokens)
                batch_predictions.append(pred_text)

                # Получение референсного текста
                ref_tokens = [TRG.vocab.itos[idx] for idx in trg_seq if idx not in
                              [TRG.vocab.stoi['<s>'], TRG.vocab.stoi['</s>'], TRG.vocab.stoi['<pad>']]]
                ref_text = ' '.join(ref_tokens)
                references.append(ref_text)

                predictions.extend(batch_predictions)

                # Вычисление ROUGE
                rouge_scores = calculate_rouge(predictions, references)

    return total_loss / len(data_loader), rouge_scores

def collate_fn(batch, src_field, trg_field, device):
    src_batch, trg_batch = [], []

    for src, trg in batch:
        # Преобразование текста в индексы
        src_tensor = torch.tensor([src_field.vocab.stoi[token] for token in src],
                                  dtype=torch.long).to(device)
        trg_tensor = torch.tensor([trg_field.vocab.stoi[token] for token in trg],
                                  dtype=torch.long).to(device)

        src_batch.append(src_tensor)
        trg_batch.append(trg_tensor)

    # Применяем паддинг
    src_padded = pad_sequence(src_batch, padding_value=src_field.vocab.stoi['<pad>'])
    trg_padded = pad_sequence(trg_batch, padding_value=trg_field.vocab.stoi['<pad>'])

    return src_padded.to(device), trg_padded.to(device)


def train_model(model, train_ds, test_ds, SRC, TRG, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epochs=10):
    # Инициализация W&B
    wandb.init(project="news-summarization",
               config={
                   "architecture": "Transformer",
                   "d_model": model.d_model,
                   "nhead": model.nhead,
                   "batch_size": 32,
                   "learning_rate": 0.0001
               })

    # Создание DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, SRC, TRG, device))
    test_loader = DataLoader(test_ds, batch_size=32,
                             collate_fn=lambda b: collate_fn(b, SRC, TRG, device))

    criterion = LabelSmoothingLoss(
        num_classes=len(TRG.vocab),  # Изменили 'classes' → 'num_classes'
        padding_idx=TRG.vocab.stoi['<pad>'],
        smoothing=0.1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    first_batch = next(iter(train_loader))
    print("\nFirst batch check:")
    print("Src:", first_batch[0].shape, "Trg:", first_batch[1].shape)
    print("Sample Src:", [SRC.vocab.itos[i] for i in first_batch[0][:5, 0]])
    print("Sample Trg:", [TRG.vocab.itos[i] for i in first_batch[1][:5, 0]])
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (src, trg) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(src, trg[:-1])
            loss = criterion(output.view(-1, output.shape[-1]),
                             trg[1:].view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Логируем каждые 50 батчей
            if batch_idx % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })

        # Средний лосс за эпоху
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}, Loss: {avg_loss}')

        # Валидация и логирование
        val_loss, rouge_scores = evaluate(model, test_loader, criterion, device, SRC, TRG)
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL']
        })

    wandb.finish()


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg in data_loader:
            output = model(src, trg[:-1])
            loss = criterion(output.view(-1, output.shape[-1]),
                             trg[1:].view(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)


def log_samples(model, SRC, TRG, device, num_samples=3):
    samples = []
    test_texts = [
        "Компания Apple представила новый iPhone",
        "Президент подписал указ о новых налогах",
        "В Москве прошёл митинг"
    ]

    for text in test_texts[:num_samples]:
        summary = generate_summary(model, SRC, TRG, text, device)
        samples.append([text, summary])

    # Создаем таблицу W&B
    table = wandb.Table(columns=["Input", "Generated Summary"], data=samples)
    wandb.log({"examples": table})