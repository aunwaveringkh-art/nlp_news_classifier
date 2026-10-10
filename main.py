"""
ФИНАЛЬНЫЙ ПАЙПЛАЙН:
TF-IDF(word+char) + SVD → LogReg + SVM + ComplementNB + LogReg(SVD)
    → Специалист для {society, politics, conflicts}
    → Финальное гибридное предсказание
"""

import os
import time
import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.data_loader import (
    load_data, preprocess_data, split_data, vectorize_data, CATEGORIES
)
from src.train import (
    train_logistic_regression, train_logistic_regression_svd,
    train_svm_calibrated, train_complement_nb,
    find_best_weights, predict_ensemble,
)
from src.stacking import (
    train_specialist, predict_with_specialist, PROBLEM_CLASSES,
)
from src.evaluate import analyze_errors
from src.utils import plot_confusion_matrix, save_metrics


def main():
    start = time.time()
    print("=" * 80)
    print("ФИНАЛЬНЫЙ ПАЙПЛАЙН: АНСАМБЛЬ + СПЕЦИАЛИСТ")
    print("=" * 80)
    
    # ============ 1. ДАННЫЕ ============
    df = load_data()
    df = preprocess_data(df, min_length=20)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # ============ 2. ВЕКТОРИЗАЦИЯ ============
    (X_train_sp, X_val_sp, X_test_sp,
     X_train_svd, X_val_svd, X_test_svd,
     vectorizers) = vectorize_data(X_train, X_val, X_test)
    
    # ============ 3. БАЗОВЫЕ МОДЕЛИ ============
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
    print("=" * 80)
    
    lr_sparse, lr_m = train_logistic_regression(
        X_train_sp, y_train, X_val_sp, y_val, C=0.5
    )
    svm_cal, svm_m = train_svm_calibrated(
        X_train_sp, y_train, X_val_sp, y_val, C=0.5
    )
    cnb, cnb_m = train_complement_nb(
        X_train_sp, y_train, X_val_sp, y_val, alpha=0.3
    )
    lr_svd, lr_svd_m = train_logistic_regression_svd(
        X_train_svd, y_train, X_val_svd, y_val, C=1.0
    )
    
    base_models = [lr_sparse, svm_cal, cnb, lr_svd]
    X_val_list = [X_val_sp, X_val_sp, X_val_sp, X_val_svd]
    X_test_list = [X_test_sp, X_test_sp, X_test_sp, X_test_svd]
    
    # ============ 4. СПЕЦИАЛИСТ ============
    print("\n" + "=" * 80)
    print("СПЕЦИАЛИСТ ДЛЯ {society, politics, conflicts}")
    print("=" * 80)
    
    specialist, spec_m = train_specialist(
        X_train_sp, y_train, X_val_sp, y_val, C=1.0
    )
    
    # ============ 5. ПОДБОР ВЕСОВ АНСАМБЛЯ ============
    weights, ens_val_acc = find_best_weights(
        base_models, X_val_list, y_val
    )
    
    # ============ 6. СРАВНЕНИЕ ВАРИАНТОВ НА ВАЛИДАЦИИ ============
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ВАРИАНТОВ НА ВАЛИДАЦИИ")
    print("=" * 80)
    
    # Вариант 1: SVM baseline
    y_val_svm = svm_cal.predict(X_val_sp)
    svm_val_acc = accuracy_score(y_val, y_val_svm)
    print(f"\n[VAL] SVM baseline:              {svm_val_acc:.4f}")
    
    # Вариант 2: Ансамбль
    y_val_ens = predict_ensemble(base_models, X_val_list, weights)
    ens_val = accuracy_score(y_val, y_val_ens)
    print(f"[VAL] Ансамбль:                  {ens_val:.4f}")
    
    # Вариант 3: SVM + Специалист
    y_val_hybrid, _, _ = predict_with_specialist(
        svm_cal, specialist, X_val_sp
    )
    hybrid_val = accuracy_score(y_val, y_val_hybrid)
    print(f"[VAL] SVM + Специалист:          {hybrid_val:.4f}")
    
    # Выбор лучшего по валидации
    candidates = {
        'svm': svm_val_acc,
        'ensemble': ens_val,
        'svm+specialist': hybrid_val,
    }
    best_name = max(candidates, key=candidates.get)
    print(f"\n[INFO] Лучший по валидации: {best_name} ({candidates[best_name]:.4f})")
    
    # ============ 7. ОЦЕНКА НА ТЕСТЕ ============
    print("\n" + "=" * 80)
    print("ОЦЕНКА НА ТЕСТЕ")
    print("=" * 80)
    
    y_test_svm = svm_cal.predict(X_test_sp)
    svm_test_acc = accuracy_score(y_test, y_test_svm)
    
    y_test_ens = predict_ensemble(base_models, X_test_list, weights)
    ens_test_acc = accuracy_score(y_test, y_test_ens)
    
    y_test_hybrid, _, _ = predict_with_specialist(
        svm_cal, specialist, X_test_sp
    )
    hybrid_test_acc = accuracy_score(y_test, y_test_hybrid)
    
    print(f"\n[RESULT] SVM baseline:           Test Acc = {svm_test_acc:.4f}")
    print(f"[RESULT] Ансамбль:               Test Acc = {ens_test_acc:.4f}")
    print(f"[RESULT] SVM + Специалист:       Test Acc = {hybrid_test_acc:.4f}")
    
    # Финальное предсказание
    if best_name == 'svm':
        final_pred = y_test_svm
    elif best_name == 'ensemble':
        final_pred = y_test_ens
    else:  # svm+specialist
        final_pred = y_test_hybrid
    
    final_acc = accuracy_score(y_test, final_pred)
    final_f1 = f1_score(y_test, final_pred, average='macro')
    
    print("\n" + "=" * 80)
    print(f"ФИНАЛЬНЫЙ ВЫБОР: {best_name}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test Macro-F1: {final_f1:.4f}")
    print("=" * 80)
    
    labels = [CATEGORIES[i] for i in sorted(CATEGORIES.keys())]
    print("\n" + classification_report(y_test, final_pred, target_names=labels, digits=4))
    
    # Матрица ошибок
    plot_confusion_matrix(y_test, final_pred, labels, save_path="confusion_matrix_final.png")
    
    # Анализ ошибок
    errors_df = analyze_errors(X_test, y_test, final_pred, n_examples=5)
    os.makedirs("results", exist_ok=True)
    errors_df.to_csv("results/errors_final.csv", index=False, encoding="utf-8")
    
    # Сохранение моделей
    joblib.dump(lr_sparse, "results/lr_sparse.pkl")
    joblib.dump(svm_cal, "results/svm_cal.pkl")
    joblib.dump(cnb, "results/cnb.pkl")
    joblib.dump(lr_svd, "results/lr_svd.pkl")
    joblib.dump(specialist, "results/specialist.pkl")
    joblib.dump(vectorizers, "results/vectorizers.pkl")
    joblib.dump(weights, "results/ensemble_weights.pkl")
    
    save_metrics({
        'svm_test_acc': float(svm_test_acc),
        'ensemble_test_acc': float(ens_test_acc),
        'svm_specialist_test_acc': float(hybrid_test_acc),
        'best_variant': best_name,
        'final_test_accuracy': float(final_acc),
        'final_test_macro_f1': float(final_f1),
        'ensemble_weights': weights.tolist(),
    }, "final_metrics.json")
    
    print("\n" + "=" * 80)
    print(f"ГОТОВО. Общее время: {time.time() - start:.1f} сек")
    print("=" * 80)


if __name__ == "__main__":
    main()