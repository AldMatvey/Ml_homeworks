from data_preparation import prepare_data, create_fields, NewsDataset
from model import TransformerModel
from train import log_samples
from train import train_model
from summarize import beam_search
from summarize import greedy_decoding
from summarize import spacy_summary
import torch
import wandb
import re
from visualize import AttentionVisualizer
from transformers import AutoModelForSeq2SeqLM


def save_model(model, SRC, TRG, path="../models"):
    import os
    os.makedirs(path, exist_ok=True)

    # Сохраняем веса модели
    torch.save(model.state_dict(), f"{path}/model_weights.pth")

    # Сохраняем дополнительную информацию
    torch.save({
        'src_vocab': SRC.vocab,
        'trg_vocab': TRG.vocab,
        'src_pad_idx': model.src_pad_idx,
        'trg_pad_idx': model.trg_pad_idx,
        'model_params': {
            'd_model': model.d_model,
            'nhead': model.nhead,
            'num_encoder_layers': len([layer for layer in model.transformer.encoder.layers]),
            'num_decoder_layers': len([layer for layer in model.transformer.decoder.layers])
        }
    }, f"{path}/model_info.pth")

    print(f"Модель сохранена в папку {path}")


def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(f"{path}/model_info.pth", map_location=device, weights_only=False)

    # Создаем экземпляр модели
    model = TransformerModel(
        vocab_sizes={
            'src': len(model_info['src_vocab']),
            'trg': len(model_info['trg_vocab'])
        },
        d_model=model_info['model_params']['d_model'],
        nhead=model_info['model_params']['nhead'],
        num_encoder_layers=model_info['model_params']['num_encoder_layers'],
        num_decoder_layers=model_info['model_params']['num_decoder_layers']
    ).to(device)

    # Загружаем веса
    model.load_state_dict(torch.load(f"{path}/model_weights.pth", map_location=device))
    model.to(device)

    # Восстанавливаем специальные токены
    model.src_pad_idx = model_info['src_pad_idx']
    model.trg_pad_idx = model_info['trg_pad_idx']

    # Восстанавливаем поля (для совместимости)
    from torchtext.data import Field
    SRC = Field(tokenize=lambda x: x.split())
    TRG = Field(tokenize=lambda x: x.split())
    SRC.vocab = model_info['src_vocab']
    TRG.vocab = model_info['trg_vocab']

    return model, SRC, TRG

def main(train = False, debug_mode = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train:
        # 1. Подготовка данных
        frac = 0.1
        if debug_mode:
            frac = 0.01
        train_df, test_df = prepare_data('../data/news.csv', sample_frac=frac)
        SRC, TRG = create_fields()

        # 2. Создание объектов Dataset
        train_data = NewsDataset(train_df, SRC, TRG)
        test_data = NewsDataset(test_df, SRC, TRG)

        # 3. Построение словарей
        SRC.build_vocab(train_data, min_freq=1)
        TRG.build_vocab(train_data, min_freq=1)
        print("\n=== Vocabulary Checks ===")
        print(f"SRC vocab size: {len(SRC.vocab)}")
        print(f"TRG vocab size: {len(TRG.vocab)}")
        print("10 most common SRC words:", SRC.vocab.freqs.most_common(10))
        print("Special tokens SRC:", {k: v for k, v in SRC.vocab.stoi.items() if k in ['<s>', '</s>', '<pad>', '<unk>']})
        print("Special tokens TRG:", {k: v for k, v in TRG.vocab.stoi.items() if k in ['<s>', '</s>', '<pad>', '<unk>']})
        # 4. Инициализация модели

        model = TransformerModel(
            vocab_sizes={'src': len(SRC.vocab), 'trg': len(TRG.vocab)},
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        ).to(device)

        # 5. Установка индексов специальных токенов
        model.src_pad_idx = SRC.vocab.stoi['<pad>']
        model.trg_pad_idx = TRG.vocab.stoi['<pad>']

        # 6. Обучение модели
        wandb.init()
        wandb.config.update({
            "src_vocab_size": len(SRC.vocab),
            "trg_vocab_size": len(TRG.vocab),
            "dataset_size": len(train_data)
        })
        # Базовый эксперимент без эмбеддингов
        model_base = TransformerModel(
            vocab_sizes={'src': len(SRC.vocab), 'trg': len(TRG.vocab)},
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        ).to(device)
        model_base.src_pad_idx = SRC.vocab.stoi['<pad>']
        model_base.trg_pad_idx = TRG.vocab.stoi['<pad>']
        train_model(
            model=model_base,
            train_ds=train_data,
            test_ds=test_data,
            SRC=SRC,
            TRG=TRG,
            device=device,
            epochs=30
        )

        # Эксперимент с предобученными эмбеддингами

        model_pretrained = TransformerModel(
            vocab_sizes={'src': len(SRC.vocab), 'trg': len(TRG.vocab)},
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            pretrained_embeddings_path='../models/cc.ru.300.bin/cc.ru.300.bin',
            SRC=SRC,
            TRG=TRG
        ).to(device)
        model_pretrained.src_pad_idx = SRC.vocab.stoi['<pad>']
        model_pretrained.trg_pad_idx = TRG.vocab.stoi['<pad>']
        train_model(
            model=model_pretrained,
            train_ds=train_data,
            test_ds=test_data,
            SRC=SRC,
            TRG=TRG,
            device=device,
            epochs=30
        )

        #save_model(model, SRC, TRG, path="news_summarizer")

    else: # уже есть обученная модель
        # 7. Тестирование генерации
        test_texts = [
            "по делу так называемого битцевского маньяка было собрано 14 доказательств",
            "Президент подписал указ о новых налогах",
            "В Москве прошёл митинг"
        ]
        model, SRC, TRG = load_model('news_summarizer')
        print("TRG vocab size:", len(TRG.vocab))
        print("20 most common TRG words:", TRG.vocab.freqs.most_common(20))
        print("Special tokens:", {k: v for k, v in TRG.vocab.stoi.items()
                                  if k in ['<s>', '</s>', '<pad>', '<unk>']})
        model.trg_pad_idx = TRG.vocab.stoi['<pad>']
        train_df, test_df = prepare_data('../data/news.csv')
        train_df[['text', 'title']].head(10)
        test_texts = [
            "о формулировке Артура Самуэля 1959 года, машинное обучение — это способ заставить компьютеры работать без программирования в явной форме. В общем случае, это процесс тонкой настройки обучения, постепенно улучшающий исходную случайную систему. То есть целью здесь является создание искусственного интеллекта, который сможет найти правильное решение из плохой системы тонкой настройкой параметров модели. Для этого в алгоритме машинного обучения используется множество различных подходов.Конкретно в этом проекте, основной подход к машинному обучению (machine learning algorithm, ML) основан на нейроэволюции. В этой форме машинного обучения используются эволюционные алгоритмы, такие как генетический алгоритм (genetic algorithm, GA), для обучения искусственных нейронных сетей (artificial neural networks, ANN).То есть в нашем случае можно сказать, что ML = GA + ANN",
            "Президент подписал указ о новых налогах",
            "В Москве прошёл митинг"
        ]
        model, SRC, TRG = load_model('news_summarizer')

        visualizer = AttentionVisualizer(model, SRC, TRG)

        for text in test_texts:
            print(f"\nInput text: {text}")

            # Обычная генерация
            summary = spacy_summary(text)
            print(f"Generated summary: {summary}")

            # Генерация с визуализацией внимания
            print("\nAttention visualization:")
            visualizer.visualize(text)


if __name__ == "__main__":
    main()
