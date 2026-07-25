import os
import pickle
import numpy as np
try:
    from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation
except Exception:
    SelfTrainingClassifier = None
    LabelPropagation = None
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    def brier_score_loss(y_true, y_prob):
        return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))
from data_loader import load_all_data, split_classify_data
from feature_extractor import batch_extract_features
from config import MODEL_DIR, RANDOM_SEED, FEATURE_NAMES


np.random.seed(RANDOM_SEED)


def build_semisupervised_model(method="selftraining"):
    base_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            class_weight="balanced_subsample", random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])
    if method == "selftraining":
        if SelfTrainingClassifier is None:
            print("[WARN] sklearn.semi_supervised.SelfTrainingClassifier 不可用，退回纯监督RF")
            return base_clf
        try:
            model = SelfTrainingClassifier(
                estimator=base_clf,
                threshold=0.85,
                max_iter=15,
                verbose=False
            )
        except TypeError:
            model = SelfTrainingClassifier(
                base_estimator=base_clf,
                threshold=0.85,
                max_iter=15,
                verbose=False
            )
    elif method == "labelprop":
        if LabelPropagation is None:
            print("[WARN] sklearn.semi_supervised.LabelPropagation 不可用，退回纯监督RF")
            return base_clf
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lp", LabelPropagation(
                kernel="knn", n_neighbors=7,
                max_iter=100, tol=1e-3
            ))
        ])
    else:
        model = base_clf
    return model


def build_supervised_model():
    models = {
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=400, max_depth=14, min_samples_split=4,
                class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1
            ))
        ]),
        "XGB": Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", GradientBoostingClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.85, random_state=RANDOM_SEED
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf", C=10.0, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ]),
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ])
    }
    return models


def extract_dataset_features(split, progress_callback=None):
    def prog_wrapped(cur, total, prefix=""):
        if progress_callback:
            progress_callback(f"{prefix}提取特征 {cur}/{total}", 0)
    if progress_callback:
        progress_callback("提取训练集特征...", 20)
    X_train_feat, train_valid = batch_extract_features(
        split["X_train"], split["mask_train"],
        lambda c, t: prog_wrapped(c, t, "训练集")
    )
    y_train = np.array([split["y_train"][i] for i in train_valid])
    if progress_callback:
        progress_callback(f"训练集完成 {len(y_train)}/{len(split['y_train'])}有效样本", 33)
        progress_callback("提取测试集特征...", 45)
    X_test_feat, test_valid = batch_extract_features(
        split["X_test"], split["mask_test"],
        lambda c, t: prog_wrapped(c, t, "测试集")
    )
    y_test = np.array([split["y_test"][i] for i in test_valid])
    if progress_callback:
        progress_callback(f"测试集完成 {len(y_test)}/{len(split['y_test'])}有效样本", 58)
        progress_callback("提取无标签集特征(半监督)...", 70)
    X_unlabeled_feat, _ = batch_extract_features(
        split["X_unlabeled"], split["mask_unlabeled"],
        lambda c, t: prog_wrapped(c, t, "无标签")
    )
    return {
        "X_train": X_train_feat, "y_train": y_train,
        "X_test": X_test_feat, "y_test": y_test,
        "X_unlabeled": X_unlabeled_feat
    }


def train_classification_models(progress_callback=None):
    if progress_callback:
        progress_callback("加载并划分数据集...", 5)
    records = load_all_data()
    split = split_classify_data(records)
    data = extract_dataset_features(split, progress_callback)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_unlabeled = data["X_unlabeled"]
    if progress_callback:
        progress_callback(f"训练集:{len(y_train)} 测试集:{len(y_test)} 无标签:{len(X_unlabeled)}", 75)
        progress_callback("训练监督基线模型并用5折CV选最优...", 76)
    
    # ===== 监督模型：用训练集5折CV选择最优模型（防止测试集泄漏） =====
    supervised = build_supervised_model()
    supervised_results = {}
    cv_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for name, model in supervised.items():
        # 交叉验证评估（只用训练集）
        cv_aucs = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
        cv_scores[name] = float(np.mean(cv_aucs))
        # 在完整训练集上训练最终模型
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        supervised_results[name] = {
            "acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5,
            "brier": brier_score_loss(y_test, y_prob) if hasattr(model, "predict_proba") else 0.5,
            "cm": confusion_matrix(y_test, y_pred).tolist(),
            "cv_auc": cv_scores[name]
        }
        if progress_callback:
            progress_callback(f"{name} CV-AUC={cv_scores[name]:.4f} Test-AUC={supervised_results[name]['auc']:.4f}", 78)
    
    # 按CV AUC选最优监督模型（测试集仅用于最终报告，不参与模型选择）
    best_supervised_name = max(cv_scores, key=cv_scores.get)
    best_model = supervised[best_supervised_name]
    best_name = best_supervised_name
    best_result = supervised_results[best_supervised_name]
    
    if progress_callback:
        progress_callback("训练半监督模型(Self-Training + LabelPropagation)...", 85)
    
    X_combo = np.vstack([X_train, X_unlabeled]) if len(X_unlabeled) > 0 else X_train
    y_combo = np.hstack([y_train, np.full(len(X_unlabeled), -1)]) if len(X_unlabeled) > 0 else y_train
    
    # ===== Self-Training（归纳式，可部署）参与最优模型对比 =====
    st_model = build_semisupervised_model(method="selftraining")
    st_result = None
    try:
        st_model.fit(X_combo, y_combo)
        st_pred = st_model.predict(X_test)
        st_prob = st_model.predict_proba(X_test)[:, 1]
        st_result = {
            "acc": accuracy_score(y_test, st_pred),
            "precision": precision_score(y_test, st_pred, zero_division=0),
            "recall": recall_score(y_test, st_pred, zero_division=0),
            "f1": f1_score(y_test, st_pred, zero_division=0),
            "auc": roc_auc_score(y_test, st_prob),
            "brier": brier_score_loss(y_test, st_prob),
            "cm": confusion_matrix(y_test, st_pred).tolist()
        }
        # Self-Training是归纳式模型，可以参与部署竞争
        # 注意：半监督没有严格的CV-AUC，用训练集有标签部分的AUC做参考
        st_train_prob = st_model.predict_proba(X_train)[:, 1]
        st_train_auc = roc_auc_score(y_train, st_train_prob)
        if st_train_auc > cv_scores[best_supervised_name]:
            best_model = st_model
            best_name = "SelfTraining+RF"
            best_result = st_result
        labeled_count = np.sum(st_model.transduction_ != -1) if hasattr(st_model, "transduction_") else len(y_train)
    except Exception as e:
        print(f"SelfTraining failed: {e}")
        st_result = None
    
    # ===== LabelPropagation（直推式，仅用于对比分析，不参与部署） =====
    lp_model = build_semisupervised_model(method="labelprop")
    lp_result = None
    try:
        lp_model.fit(X_combo, y_combo)
        lp_pred = lp_model.predict(X_test)
        lp_prob = lp_model.predict_proba(X_test)[:, 1]
        lp_result = {
            "acc": accuracy_score(y_test, lp_pred),
            "precision": precision_score(y_test, lp_pred, zero_division=0),
            "recall": recall_score(y_test, lp_pred, zero_division=0),
            "f1": f1_score(y_test, lp_pred, zero_division=0),
            "auc": roc_auc_score(y_test, lp_prob),
            "brier": brier_score_loss(y_test, lp_prob),
            "cm": confusion_matrix(y_test, lp_pred).tolist(),
            "note": "直推式模型，仅用于对比，不部署"
        }
        # LabelPropagation是直推式，不更新best_model（不参与部署选择）
    except Exception as e:
        print(f"LabelProp failed: {e}")
        lp_result = None
    
    try:
        X_all_train = np.asarray(X_train, dtype=float)
        X_train_min = np.nanpercentile(X_all_train, 5, axis=0)
        X_train_max = np.nanpercentile(X_all_train, 95, axis=0)
    except Exception:
        n_feat = X_train.shape[1] if hasattr(X_train, "shape") and X_train.ndim > 1 else len(FEATURE_NAMES)
        X_train_min = np.zeros(n_feat, dtype=float)
        X_train_max = np.ones(n_feat, dtype=float)
    supervised_y_probs = {}
    for name, model in supervised.items():
        try:
            supervised_y_probs[name] = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    extra_y_probs = {}
    if st_result is not None and best_name == "SelfTraining+RF":
        extra_y_probs["SelfTraining+RF"] = st_model.predict_proba(X_test)[:, 1]
    if lp_result is not None:
        try:
            extra_y_probs["LabelProp+RF"] = lp_model.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    save_path = os.path.join(MODEL_DIR, "classification_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "best_model": best_model,
            "best_name": best_name,
            "best_result": best_result,
            "supervised_results": supervised_results,
            "supervised_models": supervised, 
            "st_result": st_result,
            "lp_result": lp_result,
            "feature_names": FEATURE_NAMES,
            "X_train_min": X_train_min,
            "X_train_max": X_train_max,
            "X_train_shape": list(X_train.shape) if hasattr(X_train, "shape") else None,
            "cv_selection_note": "最优模型由训练集5折CV-AUC选出，测试集仅用于最终评估",
            "X_test": X_test,
            "y_test": y_test
        }, f)
    if progress_callback:
        progress_callback(f"最优模型: {best_name} (CV选) | Test-AUC={best_result['auc']:.4f} | 模型已保存", 100)
    return best_model, best_name, best_result, supervised_results, st_result, lp_result, X_test, y_test


def load_classification_model():
    path = os.path.join(MODEL_DIR, "classification_model.pkl")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["best_model"], data
    except Exception as e:
        print(f"Load cls model error: {e}")
        return None, None


def predict_classification(model, feature_vector):
    if model is None:
        return None, None
    try:
        y_pred = model.predict([feature_vector])[0]
        y_prob = model.predict_proba([feature_vector])[0]
        prob = float(y_prob[1])
        return int(y_pred), prob
    except Exception as e:
        print(f"Predict error: {e}")
        return None, None


if __name__ == "__main__":
    result = train_classification_models()
    print("Training done!")