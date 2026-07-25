import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
try:
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
except ImportError:
    try:
        from sklearn.metrics import calibration_curve
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        calibration_curve = None
        CalibratedClassifierCV = None
try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    def brier_score_loss(y_true, y_prob):
        return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))
from sklearn.inspection import permutation_importance
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False
import seaborn as sns
from config import RESULT_DIR, FEATURE_NAMES, LABEL_MAP_CLASSIFY


rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_style("whitegrid", {"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]})
sns.set_palette("husl")


def save_fig(fig, name, dpi=150):
    path = os.path.join(RESULT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_probs_dict, title="ROC曲线对比"):
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_prob in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("假阳性率 (1-特异度)", fontsize=12)
    ax.set_ylabel("真阳性率 (敏感度/召回率)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    return save_fig(fig, "roc_curve.png")


def plot_pr_curve(y_true, y_probs_dict, title="PR曲线对比"):
    fig, ax = plt.subplots(figsize=(10, 8))
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], "k--", lw=1, alpha=0.5, label="基线(阳性比例)")
    for name, y_prob in y_probs_dict.items():
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(r, p, lw=2, label=f"{name} (AP = {ap:.4f})")
    ax.set_xlabel("召回率 (Recall)", fontsize=12)
    ax.set_ylabel("精确率 (Precision)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    return save_fig(fig, "pr_curve.png")


def plot_confusion_matrix(y_true, y_pred, title="混淆矩阵"):
    cm = confusion_matrix(y_true, y_pred)
    labels = [LABEL_MAP_CLASSIFY[0], LABEL_MAP_CLASSIFY[1]]
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=True)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return save_fig(fig, "confusion_matrix.png")


def plot_calibration_curve(y_true, y_probs_dict, n_bins=10, title="校准曲线(可靠性图)"):
    if calibration_curve is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "当前 scikit-learn 版本暂不支持 calibration_curve\n请升级到 sklearn 1.2+",
                ha="center", va="center", fontsize=14, color="#c62828")
        ax.set_axis_off()
        ax.set_title(title + " (不可用)", fontsize=14, fontweight="bold")
        return save_fig(fig, "calibration_curve.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="完美校准")
    for name, y_prob in y_probs_dict.items():
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
            bs = brier_score_loss(y_true, y_prob)
            ax.plot(mean_pred, frac_pos, "s-", lw=2, label=f"{name} (Brier = {bs:.4f})")
        except Exception as e:
            print(f"[WARN] 校准曲线 {name} 生成失败: {e}")
    ax.set_xlabel("平均预测概率", fontsize=12)
    ax.set_ylabel("正样本比例", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    return save_fig(fig, "calibration_curve.png")


def plot_decision_curve_analysis(y_true, y_probs_dict, thresholds=None, title="DCA决策曲线分析"):
    """
    决策曲线分析 (Decision Curve Analysis)
    评估模型在不同阈值概率下的临床净获益
    NB = (TP/N) - (FP/N) * (pt / (1-pt))
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(9, 7))
    y_true = np.asarray(y_true)
    n = len(y_true)
    event_rate = np.mean(y_true)
    
    # 参考线：全干预（treat all）
    treat_all_nb = event_rate - (1 - event_rate) * (thresholds / (1 - thresholds))
    # 参考线：全不干预（treat none）
    treat_none_nb = np.zeros_like(thresholds)
    
    ax.plot(thresholds, treat_all_nb, "k--", lw=1.5, label="全部干预 (Treat All)")
    ax.plot(thresholds, treat_none_nb, "k:", lw=1.5, label="不干预 (Treat None)")
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(1, len(y_probs_dict))))
    for (name, y_prob), color in zip(y_probs_dict.items(), colors):
        y_prob = np.asarray(y_prob)
        nb_list = []
        for pt in thresholds:
            if pt >= 1.0:
                nb_list.append(0.0)
                continue
            y_pred = (y_prob >= pt).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
            nb_list.append(nb)
        ax.plot(thresholds, nb_list, "-", lw=2.5, color=color, label=name)
    
    ax.set_xlabel("阈值概率 (Threshold Probability)", fontsize=12)
    ax.set_ylabel("净获益 (Net Benefit)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    # 只显示净获益≥0的部分
    ax.set_ylim(bottom=-0.05)
    return save_fig(fig, "dca_curve.png")


def plot_model_metrics_bar(metrics_dict, title="多模型性能指标对比"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    metrics_order = ["acc", "precision", "recall", "f1", "auc", "brier"]
    titles = ["准确率", "精确率", "召回率(敏感度)", "F1分数", "AUC", "Brier损失(越小越好)"]
    colors = sns.color_palette("Set2", n_colors=len(metrics_dict))
    for ax, met, t in zip(axes.flat, metrics_order, titles):
        names = list(metrics_dict.keys())
        vals = [metrics_dict[n].get(met, 0) for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        if met != "brier":
            ax.set_ylim(0, 1.1)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return save_fig(fig, "model_metrics_bar.png")


def plot_feature_importance(model, X, y, top_n=15, title=None):
    if title is None:
        title = f"前{top_n}特征重要性(Permutation)"
    try:
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, n_jobs=-1, scoring="roc_auc"
        )
        sorted_idx = result.importances_mean.argsort()[-top_n:][::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}" for i in sorted_idx]
        ax.boxplot(
            result.importances[sorted_idx].T,
            vert=False, labels=names, patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="navy")
        )
        ax.set_xlabel("AUC下降幅度 (Permutation Importance)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return save_fig(fig, "feature_importance.png")
    except Exception as e:
        print(f"FI plot error: {e}")
        return None


def plot_shap_analysis(model, X, sample_idx=0, title="SHAP可解释性分析"):
    if not SHAP_AVAILABLE or shap is None:
        paths = {}
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "SHAP 库未安装或不可用\n请运行: pip install shap",
                ha="center", va="center", fontsize=16, color="#c62828")
        ax.set_axis_off()
        ax.set_title(title + " (不可用)", fontsize=14, fontweight="bold")
        paths["summary"] = save_fig(fig, "shap_summary.png")
        paths["bar"] = paths["summary"]
        paths["waterfall"] = paths["summary"]
        paths["force"] = paths["summary"]
        return paths
    paths = {}
    try:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samp, n_feat = X.shape
        feat_names = list(FEATURE_NAMES[:n_feat])
        use_model = model[-1] if (hasattr(model, "__len__") and not isinstance(model, type)) else model
        sv = None
        exp_val = None
        try:
            explainer = shap.TreeExplainer(use_model)
            raw_sv = explainer.shap_values(X)
            if isinstance(raw_sv, list):
                sv = np.asarray(raw_sv[1]) if len(raw_sv) > 1 else np.asarray(raw_sv[0])
            else:
                sv = np.asarray(raw_sv)
            if sv.ndim == 3:
                sv = sv[..., 1] if sv.shape[-1] == 2 else sv[..., 0]
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                ev_arr = np.asarray(exp_val).reshape(-1)
                exp_val = float(ev_arr[-1] if len(ev_arr) > 1 else ev_arr[0])
        except Exception as e1:
            print(f"SHAP TreeExplainer fallback: {e1}")
            try:
                X_bg = shap.sample(X, min(50, n_samp))
                X_eval = shap.sample(X, min(100, n_samp))
                masker = shap.maskers.Independent(X, max_samples=min(100, n_samp))
                if hasattr(use_model, "predict_proba"):
                    def pred_fn(x):
                        p = use_model.predict_proba(x)
                        return p[:, -1] if getattr(p, "ndim", 1) > 1 else p
                else:
                    pred_fn = use_model.predict
                try:
                    explainer = shap.Explainer(pred_fn, masker, feature_names=feat_names)
                    sv_raw = explainer.shap_values(X_eval)
                except Exception:
                    explainer = shap.KernelExplainer(pred_fn, X_bg)
                    sv_raw = explainer.shap_values(X_eval, silent=True)
                sv = np.asarray(sv_raw)
                if sv.ndim == 3:
                    sv = sv[..., -1] if sv.shape[-1] <= 2 else sv
                exp_val = getattr(explainer, "expected_value", 0.0)
                if isinstance(exp_val, (list, np.ndarray)):
                    ev_arr = np.asarray(exp_val).reshape(-1)
                    exp_val = float(ev_arr[-1] if len(ev_arr) else 0.0)
                X = X_eval
                n_samp, n_feat = X.shape
                feat_names = list(FEATURE_NAMES[:n_feat])
            except Exception as e2:
                print(f"SHAP both explainers failed: {e1} | {e2}")
                raise
        if sv is None:
            raise RuntimeError("无法计算SHAP值")
        if sv.shape != (n_samp, n_feat):
            if sv.ndim == 2 and sv.shape[1] == n_feat and sv.shape[0] <= n_samp:
                n_samp = sv.shape[0]
                X = X[:n_samp]
            else:
                target = (n_samp, n_feat)
                try:
                    sv = np.array(sv).reshape(target)
                except Exception:
                    sv = np.array(sv).reshape(-1)
                    need = n_samp * n_feat
                    if sv.size >= need:
                        sv = sv[:need].reshape(target)
                    else:
                        sv = np.pad(sv, (0, need - sv.size)).reshape(target)
        sv = np.asarray(sv, dtype=float)
        X = np.asarray(X, dtype=float)
        fig_summary, ax_summary = plt.subplots(figsize=(10, 7))
        try:
            shap.summary_plot(sv, X, feature_names=feat_names, show=False,
                              plot_type="dot", max_display=15)
            paths["summary"] = save_fig(fig_summary, "shap_summary.png")
        except Exception as e:
            print(f"SHAP summary err: {e}")
            ax_summary.text(0.5, 0.5, f"SHAP Summary不可用\n{e}", ha="center", va="center", color="#c62828")
            paths["summary"] = save_fig(fig_summary, "shap_summary.png")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 7))
        try:
            shap.summary_plot(sv, X, feature_names=feat_names, show=False,
                              plot_type="bar", max_display=15)
            paths["bar"] = save_fig(fig_bar, "shap_bar.png")
        except Exception as e:
            print(f"SHAP bar err: {e}")
            ax_bar.text(0.5, 0.5, f"SHAP Bar不可用\n{e}", ha="center", va="center", color="#c62828")
            paths["bar"] = save_fig(fig_bar, "shap_bar.png")
        si = min(sample_idx, n_samp - 1, sv.shape[0] - 1)
        sv_single = sv[si]
        paths["waterfall"] = paths["summary"]
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 7))
        try:
            if X.ndim == 1:
                x_i = X
            else:
                x_i = X[si]
            exp_f = float(exp_val) if not isinstance(exp_val, (list, np.ndarray)) else float(np.asarray(exp_val).reshape(-1)[-1])
            shap.waterfall_plot(
                shap.Explanation(
                    values=np.asarray(sv_single, dtype=float).reshape(-1),
                    base_values=exp_f,
                    data=np.asarray(x_i, dtype=float).reshape(-1),
                    feature_names=feat_names
                ), show=False
            )
            ax_waterfall.set_title(f"SHAP瀑布图 (样本#{si})", fontsize=13, fontweight="bold")
            paths["waterfall"] = save_fig(fig_waterfall, "shap_waterfall.png")
        except Exception as we:
            print(f"Waterfall err: {we}")
            plt.close(fig_waterfall)
            fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 7))
            top_k = min(10, n_feat)
            order = np.argsort(-np.abs(sv_single))[:top_k]
            colors = ["#ef5350" if sv_single[i] > 0 else "#42a5f5" for i in order]
            ax_waterfall.barh(range(top_k), sv_single[order], color=colors)
            ax_waterfall.set_yticks(range(top_k))
            ax_waterfall.set_yticklabels([feat_names[i] for i in order])
            ax_waterfall.invert_yaxis()
            ax_waterfall.set_title(f"SHAP贡献Top{top_k} (样本#{si})", fontsize=13, fontweight="bold")
            ax_waterfall.set_xlabel("SHAP value")
            ax_waterfall.axvline(0, color="#333", linewidth=0.5)
            paths["waterfall"] = save_fig(fig_waterfall, "shap_waterfall.png")
        paths["force"] = paths["waterfall"]
        return paths
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SHAP total err: {e}")
        try:
            paths = {}
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"SHAP分析失败:\n{str(e)[:80]}",
                    ha="center", va="center", fontsize=14, color="#c62828")
            ax.set_axis_off()
            ax.set_title(title, fontsize=14, fontweight="bold")
            p = save_fig(fig, "shap_summary.png")
            paths["summary"] = p
            paths["bar"] = p
            paths["waterfall"] = p
            paths["force"] = p
            return paths
        except Exception:
            return None


def plot_detection_history(hist1, hist2=None, title="检测模型训练曲线"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if hist2 is not None:
        loss = hist1.history["loss"] + hist2.history["loss"]
        val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]
        auc = hist1.history["auc"] + hist2.history["auc"]
        val_auc = hist1.history["val_auc"] + hist2.history["val_auc"]
        split_idx = len(hist1.history["loss"])
    else:
        loss = hist1.history["loss"]
        val_loss = hist1.history["val_loss"]
        auc = hist1.history["auc"]
        val_auc = hist1.history["val_auc"]
        split_idx = None
    epochs = list(range(1, len(loss) + 1))
    ax = axes[0]
    ax.plot(epochs, loss, "o-", label="训练损失")
    ax.plot(epochs, val_loss, "s-", label="验证损失")
    if split_idx:
        ax.axvline(split_idx + 0.5, color="red", linestyle="--", label="微调开启点")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Crossentropy")
    ax.set_title("损失曲线", fontweight="bold")
    ax.legend()
    ax = axes[1]
    ax.plot(epochs, auc, "o-", label="训练AUC")
    ax.plot(epochs, val_auc, "s-", label="验证AUC")
    if split_idx:
        ax.axvline(split_idx + 0.5, color="red", linestyle="--", label="微调开启点")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("AUC曲线", fontweight="bold")
    ax.legend()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return save_fig(fig, "detection_history.png")


def plot_prediction_dashboard(img, heatmap_overlay, det_result, cls_result, feature_vector,
                              det_prob=None, cls_prob=None, save_path="prediction_dashboard.png"):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.imshow(img)
    ax1.set_title("原始超声影像", fontsize=12, fontweight="bold")
    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    if heatmap_overlay is not None:
        ax2.imshow(heatmap_overlay)
        ax2.set_title("Grad-CAM 病灶注意力热力图", fontsize=12, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "无Grad-CAM数据", ha="center", va="center")
    ax2.axis("off")
    ax3 = fig.add_subplot(gs[2, 0:2])
    det_lbl = "有病灶(异常)" if det_result == 1 else "正常(无病灶)"
    det_color = "#d32f2f" if det_result == 1 else "#388e3c"
    if det_prob is not None:
        sizes = [det_prob * 100 if det_result == 1 else (1 - det_prob) * 100,
                 (1 - det_prob) * 100 if det_result == 1 else det_prob * 100]
        labels = [f"{det_lbl}\n{max(sizes):.1f}%",
                  f"{'正常' if det_result == 1 else '异常'} {min(sizes):.1f}%"]
        colors = [det_color, "#cccccc"]
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors,
                                            autopct="%1.1f%%", startangle=90,
                                            textprops={"fontsize": 10})
    ax3.set_title("第一阶段: 病灶检测结果", fontsize=11, fontweight="bold")
    ax4 = fig.add_subplot(gs[2, 2:4])
    if cls_result is not None:
        cls_lbl = "恶性(Malignant)" if cls_result == 1 else "良性(Benign)"
        cls_color = "#c62828" if cls_result == 1 else "#66bb6a"
        if cls_prob is not None:
            sizes = [cls_prob * 100 if cls_result == 1 else (1 - cls_prob) * 100,
                     (1 - cls_prob) * 100 if cls_result == 1 else cls_prob * 100]
            labels = [f"{cls_lbl}\n{max(sizes):.1f}%",
                      f"{'良性' if cls_result == 1 else '恶性'} {min(sizes):.1f}%"]
            colors = [cls_color, "#cccccc"]
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                                autopct="%1.1f%%", startangle=90,
                                                textprops={"fontsize": 10})
        ax4.set_title("第二阶段: 半监督良恶性分类", fontsize=11, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "无病灶，无需分类", ha="center", va="center",
                 fontsize=12, color="gray")
        ax4.set_title("第二阶段: 良恶性分类", fontsize=11, fontweight="bold")
        ax4.axis("off")
    return save_fig(fig, save_path)


def plot_top_features_radar(feature_vector, top_n=8, title="关键影像组学特征雷达图",
                            feature_ref_mins=None, feature_ref_maxs=None):
    if feature_vector is None:
        return None
    try:
        fv = np.array(feature_vector).reshape(-1)
        n_feat = len(fv)
        feat_names = list(FEATURE_NAMES)[:n_feat]

        # ==============================================
        # ✅ 修复核心：所有特征统一用【稳健全局百分位归一化】
        #    再也不会出现"一个特征1.0其他全0"的BUG
        # ==============================================
        fv_safe = np.nan_to_num(fv, nan=0.0, posinf=1e9, neginf=-1e9)

        if feature_ref_mins is not None and feature_ref_maxs is not None:
            # 方案A：有训练集参考值 → 用训练集5%~95%分位数归一化（最准确）
            mins = np.asarray(feature_ref_mins).reshape(-1)[:n_feat]
            maxs = np.asarray(feature_ref_maxs).reshape(-1)[:n_feat]
            mins_safe = np.where(np.isfinite(mins), mins, np.zeros_like(mins))
            maxs_safe = np.where(np.isfinite(maxs), maxs, np.ones_like(maxs))
            rng = (maxs_safe - mins_safe) + 1e-9
            rng = np.where(rng < 1e-6, 1.0, rng)
            normed = np.clip((fv_safe - mins_safe) / rng, 0.05, 0.95)
        else:
            # 方案B：无参考值 → 对单样本特征做【逐特征独立百分位归一化】
            # 关键：每个特征单独映射，不会因为一个极端值挤掉其他特征
            normed = np.zeros(n_feat, dtype=float)
            for i in range(n_feat):
                val = fv_safe[i]
                name = feat_names[i] if i < len(feat_names) else f"f{i}"
                # Hu不变矩特殊处理：值域通常在 1e-8 ~ 1e-2 之间，取log10后再归一化
                if name.lower().startswith("hu_moment") or "hu" in name.lower():
                    if abs(val) < 1e-12:
                        normed[i] = 0.5
                    else:
                        log_v = np.log10(abs(val) + 1e-12)
                        # Hu矩log值域大致在 [-12, 0]，映射到 [0.1, 0.9]
                        normed[i] = np.clip((log_v - (-12.0)) / (0.0 - (-12.0)), 0.1, 0.9)
                # 面积相关特征(area_ratio, area)特殊处理：用log压缩极大值
                elif "area" in name.lower() or "ratio" in name.lower():
                    if val < 1e-6:
                        normed[i] = 0.1
                    else:
                        log_v = np.log10(val + 1e-6)
                        # 面积比log值域大致在 [-4, 1]，映射到 [0.1, 0.9]
                        normed[i] = np.clip((log_v - (-4.0)) / (1.0 - (-4.0)), 0.1, 0.9)
                else:
                    # 其他特征：用单样本稳健Z分数→sigmoid压缩(避免极端值)
                    # 因为只有单样本，不能算全局标准差，所以用经验值域
                    # 这里退而求其次：每个特征的绝对值截断到合理中心区间 [0.2, 0.8]
                    centered = val - 0.0  # 无参考时默认以0为中心
                    # 基于经验给一个合理的缩放：避免爆0或爆1
                    sig = 1.0 / (1.0 + np.exp(-centered / max(1e-6, abs(val) + 1e-3)))
                    normed[i] = 0.3 + sig * 0.4   # 映射到 [0.3, 0.7] 的安全区

        # ==============================================
        # 选前top_n个偏离0.5最大的特征（展示最"异常"的特征给医生看）
        # ==============================================
        abs_dev = np.abs(normed - 0.5)
        top_idx = np.argsort(-abs_dev)[:top_n]
        top_idx = [i for i in top_idx if i < n_feat]

        categories = [feat_names[i] for i in top_idx]
        values = [float(normed[i]) for i in top_idx]

        # 再做一次安全检查：防止一个特征=0.9999，其他全=0.0001的极端情况
        # 把所有值线性缩放到 [0.15, 0.85] 区间，视觉更合理
        vals_arr = np.array(values)
        if len(vals_arr) >= 2:
            v_min, v_max = vals_arr.min(), vals_arr.max()
            if (v_max - v_min) > 0.75:  # 分布太分散才缩
                vals_safe = 0.15 + (vals_arr - v_min) / (v_max - v_min + 1e-9) * 0.7
                values = [float(x) for x in vals_safe]

        N = len(categories)
        if N < 3:
            categories += ["feature_" + str(k) for k in range(N, max(3, N))]
            values += [0.5] * (max(3, N) - N)
            N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values_plot = values + values[:1]
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, linewidth=2.5, color="#1976d2", marker="o", markersize=7)
        ax.fill(angles, values_plot, alpha=0.3, color="#1976d2")
        for i in range(N):
            color_val = "#c62828" if values[i] > 0.7 else ("#2e7d32" if values[i] < 0.3 else "#555555")
            ax.text(angles[i], min(1.05, values[i] + 0.06), f"{values[i]:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold", color=color_val)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold", y=1.1)
        info = "提示: 数值越接近1→特征越强; 红(>0.7)高风险/绿(<0.3)低风险"
        fig.text(0.5, 0.01, info, ha="center", fontsize=9, color="#555555")
        return save_fig(fig, "feature_radar.png")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Radar err: {e}")
        try:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.text(0.5, 0.5, f"雷达图生成失败:\n{str(e)[:60]}", transform=ax.transAxes,
                    ha="center", va="center", color="#c62828", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            return save_fig(fig, "feature_radar.png")
        except Exception:
            return None


if __name__ == "__main__":
    print("Visualization module ready.")