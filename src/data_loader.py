"""
Модуль для загрузки и предобработки датасета rus_news_classifier.
Объединённая векторизация: word-TF-IDF (1,3) + char-TF-IDF (3,5) + SVD.
"""

import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack


CATEGORIES = {
    0: 'climate',
    1: 'conflicts',
    2: 'culture',
    3: 'economy',
    4: 'gloss',
    5: 'health',
    6: 'politics',
    7: 'science',
    8: 'society',
    9: 'sports',
    10: 'travel',
}


def load_data() -> pd.DataFrame:
    """Загружает датасет с Hugging Face."""
    print("[INFO] Загружаем датасет data-silence/rus_news_classifier...")
    dataset = load_dataset("data-silence/rus_news_classifier")
    
    dfs = []
    for split_name in dataset.keys():
        df_split = dataset[split_name].to_pandas()
        df_split['split'] = split_name
        dfs.append(df_split)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Загружено {len(df)} примеров.")
    print(f"[INFO] Распределение классов:\n{df['labels'].value_counts().sort_index()}")
    return df


def clean_text(text: str) -> str:
    """Базовая очистка текста."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(df: pd.DataFrame, min_length: int = 20) -> pd.DataFrame:
    """Очистка текста и удаление коротких примеров."""
    print("[INFO] Предобработка данных...")
    df = df.copy()
    df['clean_news'] = df['news'].apply(clean_text)
    
    before = len(df)
    df = df[df['clean_news'].str.len() >= min_length].reset_index(drop=True)
    print(f"[INFO] Удалено коротких текстов: {before - len(df)}")
    return df


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Разбивает данные на train/val/test со стратификацией."""
    print(f"[INFO] Разбиваем данные: test={test_size}, val={val_size}")
    
    X = np.array(df['clean_news'].tolist(), dtype=object)
    y = np.array(df['labels'].tolist(), dtype=np.int64)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative,
        random_state=random_state, stratify=y_temp
    )
    
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_data(X_train, X_val, X_test):
    """
    Объединённая векторизация: word-TF-IDF + char-TF-IDF + TruncatedSVD.
    Возвращает: sparse-матрицы (word+char) и dense-матрицы (SVD).
    """
    print("[INFO] Векторизация word-TF-IDF (1,3)...")
    word_vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=80000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents='unicode',
    )
    X_train_w = word_vec.fit_transform(X_train)
    X_val_w = word_vec.transform(X_val)
    X_test_w = word_vec.transform(X_test)
    print(f"[INFO] Word TF-IDF: {X_train_w.shape}")
    
    print("[INFO] Векторизация char-TF-IDF (3,5)...")
    char_vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents='unicode',
    )
    X_train_c = char_vec.fit_transform(X_train)
    X_val_c = char_vec.transform(X_val)
    X_test_c = char_vec.transform(X_test)
    print(f"[INFO] Char TF-IDF: {X_train_c.shape}")
    
    # Sparse для LogReg/SVM/NB
    X_train_sparse = hstack([X_train_w, X_train_c]).tocsr()
    X_val_sparse = hstack([X_val_w, X_val_c]).tocsr()
    X_test_sparse = hstack([X_test_w, X_test_c]).tocsr()
    print(f"[INFO] Объединённая sparse-форма train: {X_train_sparse.shape}")
    
    # TruncatedSVD (LSA)
    print("[INFO] TruncatedSVD (n_components=400)...")
    svd = TruncatedSVD(n_components=400, random_state=42, n_iter=5)
    X_train_svd = svd.fit_transform(X_train_sparse)
    X_val_svd = svd.transform(X_val_sparse)
    X_test_svd = svd.transform(X_test_sparse)
    print(f"[INFO] SVD-форма train: {X_train_svd.shape}")
    print(f"[INFO] Explained variance (sum): {svd.explained_variance_ratio_.sum():.4f}")
    
    vectorizers = {'word': word_vec, 'char': char_vec, 'svd': svd}
    return (
        X_train_sparse, X_val_sparse, X_test_sparse,
        X_train_svd, X_val_svd, X_test_svd,
        vectorizers
    )