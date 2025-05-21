import pandas as pd
import re
from torch.utils.data import Dataset
from torchtext.data import Field
from sklearn.model_selection import train_test_split


class NewsDataset(Dataset):
    def __init__(self, df, src_field, trg_field):
        self.df = df
        self.src_field = src_field
        self.trg_field = trg_field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src, trg = self.df.iloc[idx]['text'], self.df.iloc[idx]['title']
        return self.src_field.preprocess(src), self.trg_field.preprocess(trg)


def clean_text(text):
    """Функция очистки для одного текста (не для целой колонки)"""
    if not isinstance(text, str):  # На случай NaN или других не-строк
        return ""
    # Сохраняем кириллицу, цифры и основные знаки препинания
    text = re.sub(r'[^а-яё0-9\s.,!?–-]', '', text.lower())
    return text.strip()


def prepare_data(filepath, sample_frac=1.0):
    # Загрузка и очистка данных
    data = pd.read_csv(filepath)
    data = data.sample(frac=sample_frac, random_state=42)

    # Применяем clean_text к каждому элементу колонки
    data['text'] = data['text'].apply(clean_text)
    data['title'] = data['title'].apply(clean_text)

    # Проверка данных
    print("\n=== Data Checks ===")
    print("Sample texts:")
    print(data['text'].head(3))
    print("\nSample titles:")
    print(data['title'].head(3))
    print("\nLength distribution:")
    print(data['title'].str.split().apply(len).describe())

    return train_test_split(data, test_size=0.1, random_state=42)


def create_fields():
    tokenizer = lambda x: x.split()
    SRC = Field(tokenize=tokenizer, init_token='<s>', eos_token='</s>', lower=True)
    TRG = Field(tokenize=tokenizer, init_token='<s>', eos_token='</s>', lower=True)
    return SRC, TRG