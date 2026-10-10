"""
Модуль обучения 4 моделей: LogReg, Calibrated SVM, ComplementNB, LogReg на SVD.
Веса ансамбля — softmax от accuracy на валидации (все модели вносят вклад).
"""

import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score


def train_logistic_regression(X_train, y_train, X_val, y_val, C=0.5):
    """LogReg на разреженных признаках (saga)."""
    print(f"\n[INFO] LogReg (sparse, C={C})...")
    model = LogisticRegression(
        C=C, max_iter=300, solver='saga',
        penalty='l2', random_state=42
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"[INFO] Время: {dt:.1f} сек | Val Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
    return model, {'accuracy': acc, 'macro_f1': f1}


def train_logistic_regression_svd(X_train, y_train, X_val, y_val, C=1.0):
    """LogReg на плотных SVD-признаках."""
    print(f"\n[INFO] LogReg (SVD, C={C})...")
    model = LogisticRegression(
        C=C, max_iter=500, solver='lbfgs',
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"[INFO] Время: {dt:.1f} сек | Val Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
    return model, {'accuracy': acc, 'macro_f1': f1}


def train_svm_calibrated(X_train, y_train, X_val, y_val, C=0.5):
    """Калиброванный LinearSVC с вероятностями."""
    print(f"\n[INFO] Calibrated SVM (C={C})...")
    base = LinearSVC(C=C, max_iter=3000, random_state=42, class_weight='balanced')
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"[INFO] Время: {dt:.1f} сек | Val Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
    return model, {'accuracy': acc, 'macro_f1': f1}


def train_complement_nb(X_train, y_train, X_val, y_val, alpha=0.3):
    """ComplementNB — специально для пересекающихся/несбалансированных классов."""
    print(f"\n[INFO] ComplementNB (alpha={alpha})...")
    model = ComplementNB(alpha=alpha)
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"[INFO] Время: {dt:.1f} сек | Val Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
    return model, {'accuracy': acc, 'macro_f1': f1}


def find_best_weights(models, X_val_list, y_val, n_iter=200):
    """
    Подбирает веса ансамбля через softmax от accuracy на валидации.
    Это гарантирует, что ВСЕ модели вносят вклад (ансамбль реален).
    
    Args:
        models: список обученных моделей (все имеют predict_proba)
        X_val_list: список соответствующих X_val (sparse или dense)
        y_val: истинные метки
        n_iter: не используется, оставлен для совместимости
    
    Returns:
        weights: массив весов
        val_acc: accuracy ансамбля на валидации
    """
    print("\n[INFO] Расчёт весов ансамбля (softmax от accuracy)...")
    
    probas = [m.predict_proba(X) for m, X in zip(models, X_val_list)]
    n_models = len(probas)
    
    # Считаем accuracy каждой модели
    accs = np.array([
        accuracy_score(y_val, probas[i].argmax(axis=1))
        for i in range(n_models)
    ])
    
    for i, acc in enumerate(accs):
        print(f"  Модель {i}: Val Acc = {acc:.4f}")
    
    # Softmax над accuracy — чем точнее, тем больше вес
    # temperature=10 — умеренная резкость
    exp_accs = np.exp(accs * 10)
    weights = exp_accs / exp_accs.sum()
    
    print(f"[INFO] Веса (softmax): {np.round(weights, 3)}")
    
    # Проверяем accuracy ансамбля
    proba_ens = sum(weights[i] * probas[i] for i in range(n_models))
    ens_acc = accuracy_score(y_val, proba_ens.argmax(axis=1))
    print(f"[INFO] Val Accuracy ансамбля: {ens_acc:.4f}")
    
    return weights, ens_acc


def predict_ensemble(models, X_list, weights):
    """Предсказание взвешенного ансамбля."""
    probas = [m.predict_proba(X) for m, X in zip(models, X_list)]
    proba = sum(weights[i] * probas[i] for i in range(len(models)))
    return proba.argmax(axis=1)