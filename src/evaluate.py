"""
Модуль оценки моделей и анализа ошибок.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader import CATEGORIES
from src.utils import (
    plot_confusion_matrix, 
    print_classification_report,
    save_metrics
)


def evaluate_model(model, X_test, y_test, model_name: str = "model"):
    """
    Оценивает модель на тестовой выборке.
    
    Args:
        model: обученная модель
        X_test, y_test: тестовые данные
        model_name: название модели для логов
        
    Returns:
        dict с метриками
    """
    print(f"\n[INFO] Оценка {model_name} на тесте...")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    labels = [CATEGORIES[i] for i in sorted(CATEGORIES.keys())]
    
    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] Macro-F1: {f1_macro:.4f}")
    print(f"[RESULT] Weighted-F1: {f1_weighted:.4f}")
    
    # Отчёт по классам
    print_classification_report(y_test, y_pred, labels)
    
    # Матрица ошибок
    plot_confusion_matrix(y_test, y_pred, labels)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
    }
    
    return metrics, y_pred


def analyze_errors(X_test, y_test, y_pred, n_examples: int = 10):
    """
    Анализирует ошибочные предсказания.
    
    Args:
        X_test: тексты
        y_test: истинные метки
        y_pred: предсказанные метки
        n_examples: сколько примеров вывести
        
    Returns:
        DataFrame с ошибками
    """
    print(f"\n[INFO] Анализ ошибок (первые {n_examples})...")
    
    # Находим ошибки
    error_mask = y_test != y_pred
    error_indices = np.where(error_mask)[0]
    
    print(f"[INFO] Всего ошибок: {len(error_indices)} из {len(y_test)} "
          f"({100 * len(error_indices) / len(y_test):.2f}%)")
    
    # Собираем DataFrame
    errors_df = pd.DataFrame({
        'text': X_test[error_indices],
        'true_label': y_test[error_indices],
        'pred_label': y_pred[error_indices],
    })
    errors_df['true_category'] = errors_df['true_label'].map(CATEGORIES)
    errors_df['pred_category'] = errors_df['pred_label'].map(CATEGORIES)
    
    # Выводим примеры
    for i, row in errors_df.head(n_examples).iterrows():
        print("\n" + "-" * 80)
        print(f"Текст: {row['text'][:300]}...")
        print(f"Истинный класс: {row['true_category']}")
        print(f"Предсказано:    {row['pred_category']}")
    
    # Самые частые пары ошибок
    print("\n[INFO] Топ-10 самых частых ошибок (true → pred):")
    confusion_pairs = errors_df.groupby(['true_category', 'pred_category']).size()
    confusion_pairs = confusion_pairs.sort_values(ascending=False)
    print(confusion_pairs.head(10))
    
    return errors_df