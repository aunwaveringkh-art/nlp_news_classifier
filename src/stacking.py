"""
Стекинг и специализированный классификатор для проблемной тройки.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score


# Индексы проблемной тройки
PROBLEM_CLASSES = [8, 6, 1]  # society, politics, conflicts
PROBLEM_SET = set(PROBLEM_CLASSES)


def filter_problem_samples(X, y):
    """Оставляет только примеры из проблемной тройки."""
    mask = np.isin(y, PROBLEM_CLASSES)
    return X[mask], y[mask]


def train_specialist(X_train_sp, y_train, X_val_sp, y_val, C=1.0):
    """
    Специализированный классификатор для {society, politics, conflicts}.
    Обучается только на примерах этих 3 классов.
    """
    print(f"\n[INFO] Специализированный SVM для тройки {PROBLEM_CLASSES}...")
    X_tr, y_tr = filter_problem_samples(X_train_sp, y_train)
    X_va, y_va = filter_problem_samples(X_val_sp, y_val)
    
    print(f"[INFO] Примеров для специалиста: train={len(y_tr)}, val={len(y_va)}")
    print(f"[INFO] Распределение train: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    
    base = LinearSVC(C=C, max_iter=5000, random_state=42, class_weight='balanced')
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    
    t0 = time.time()
    model.fit(X_tr, y_tr)
    dt = time.time() - t0
    
    y_pred = model.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred, average='macro')
    
    print(f"[INFO] Время: {dt:.1f} сек")
    print(f"[INFO] Specialist Val Acc (на тройке): {acc:.4f}")
    print(f"[INFO] Specialist Val Macro-F1 (на тройке): {f1:.4f}")
    
    return model, {'accuracy': acc, 'macro_f1': f1}


def train_meta_classifier(base_models, X_val_list, y_val, X_test_list=None):
    """
    Стекинг: обучаем мета-LogReg поверх вероятностей базовых моделей.
    Мета-модель видит concatenated probabilities всех моделей.
    """
    print("\n[INFO] Обучение мета-классификатора (стекинг)...")
    
    # Собираем вероятности базовых моделей на валидации
    val_probas = [m.predict_proba(X) for m, X in zip(base_models, X_val_list)]
    X_meta_val = np.hstack(val_probas)
    print(f"[INFO] Размер мета-признаков: {X_meta_val.shape}")
    
    meta_model = LogisticRegression(
        C=1.0, max_iter=500, solver='lbfgs', random_state=42
    )
    t0 = time.time()
    meta_model.fit(X_meta_val, y_val)
    dt = time.time() - t0
    
    y_pred = meta_model.predict(X_meta_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"[INFO] Время обучения мета: {dt:.1f} сек")
    print(f"[INFO] Meta Val Acc (на своей же val — оптимистично): {acc:.4f}")
    print(f"[INFO] Meta Val Macro-F1: {f1:.4f}")
    
    return meta_model


def predict_with_specialist(base_model, specialist_model, X_sparse):
    """
    Гибридное предсказание:
    1. Базовая модель даёт ответ и вероятности.
    2. Если ответ из тройки — проверяем специалиста.
    3. Если специалист уверен (>0.5) — берём его ответ, иначе базовый.
    """
    base_proba = base_model.predict_proba(X_sparse)
    base_pred = base_proba.argmax(axis=1)
    
    # Специалист возвращает уже настоящие метки (1, 6, 8), не индексы
    spec_pred = specialist_model.predict(X_sparse)
    spec_proba = specialist_model.predict_proba(X_sparse)
    spec_max_proba = spec_proba.max(axis=1)
    
    # Финальное предсказание
    final_pred = base_pred.copy()
    mask_in_triple = np.isin(base_pred, PROBLEM_CLASSES)
    mask_confident_spec = spec_max_proba > 0.5
    
    # Заменяем только там, где базовый предсказал из тройки И специалист уверен
    replace = mask_in_triple & mask_confident_spec
    final_pred[replace] = spec_pred[replace]
    
    return final_pred, base_pred, spec_pred


def predict_stacking(meta_model, base_models, X_list):
    """Предсказание через мета-классификатор."""
    probas = [m.predict_proba(X) for m, X in zip(base_models, X_list)]
    X_meta = np.hstack(probas)
    return meta_model.predict(X_meta)