"""
Вспомогательные функции: сохранение моделей, графики, метрики.
"""

import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


RESULTS_DIR = "results"


def ensure_results_dir():
    """Создаёт папку results, если её нет."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_model(model, vectorizer, path_prefix: str = "svm_model"):
    """
    Сохраняет модель и векторизатор.
    
    Args:
        model: обученная модель
        vectorizer: TF-IDF векторизатор
        path_prefix: префикс имени файла
    """
    ensure_results_dir()
    model_path = os.path.join(RESULTS_DIR, f"{path_prefix}.pkl")
    vec_path = os.path.join(RESULTS_DIR, f"{path_prefix}_vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"[INFO] Модель сохранена: {model_path}")
    print(f"[INFO] Векторизатор сохранён: {vec_path}")


def load_model(path_prefix: str = "svm_model"):
    """Загружает модель и векторизатор."""
    model_path = os.path.join(RESULTS_DIR, f"{path_prefix}.pkl")
    vec_path = os.path.join(RESULTS_DIR, f"{path_prefix}_vectorizer.pkl")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def save_metrics(metrics: dict, path: str = "metrics.json"):
    """Сохраняет метрики в JSON."""
    ensure_results_dir()
    full_path = os.path.join(RESULTS_DIR, path)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Метрики сохранены: {full_path}")


def plot_confusion_matrix(y_true, y_pred, labels: list, save_path: str = "confusion_matrix.png"):
    """
    Строит и сохраняет матрицу ошибок.
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        labels: список названий классов
        save_path: путь для сохранения
    """
    ensure_results_dir()
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Количество'}
    )
    plt.title('Матрица ошибок', fontsize=14)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    full_path = os.path.join(RESULTS_DIR, save_path)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Матрица ошибок сохранена: {full_path}")


def print_classification_report(y_true, y_pred, labels: list):
    """Печатает отчёт по классам."""
    report = classification_report(
        y_true, y_pred, 
        target_names=labels, 
        digits=4
    )
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(report)
    return report