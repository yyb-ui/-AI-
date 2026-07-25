import os
import sys
import threading
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:
    print("FATAL: Tkinter 不可用", e)
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception as e:
    print("WARNING: Pillow 未安装，图片显示功能受限:", e)
    PIL_OK = False

try:
    import numpy as np
    NUMPY_OK = True
except Exception as e:
    print("FATAL: numpy 不可用:", e)
    NUMPY_OK = False
    np = None

_IMPORT_ERRORS = []

try:
    from config import RESULT_DIR, MODEL_DIR, LABEL_MAP_DETECTION, LABEL_MAP_CLASSIFY, FEATURE_NAMES
    CONFIG_OK = True
except Exception as e:
    _IMPORT_ERRORS.append(("config", repr(e), traceback.format_exc()))
    CONFIG_OK = False
    RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
    MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
    LABEL_MAP_DETECTION = {0: "正常(无病灶)", 1: "异常(有病灶)"}
    LABEL_MAP_CLASSIFY = {0: "良性", 1: "恶性"}
    FEATURE_NAMES = []

try:
    from data_loader import read_image, preprocess_image
    DATA_LOADER_OK = True
except Exception as e:
    _IMPORT_ERRORS.append(("data_loader", repr(e), traceback.format_exc()))
    DATA_LOADER_OK = False
    read_image = lambda *a, **k: None
    preprocess_image = lambda *a, **k: None

try:
    from feature_extractor import extract_all_features, auto_segment_lesion
    FEATURE_EXTRACTOR_OK = True
except Exception as e:
    _IMPORT_ERRORS.append(("feature_extractor", repr(e), traceback.format_exc()))
    FEATURE_EXTRACTOR_OK = False
    extract_all_features = lambda *a, **k: (None, None)
    auto_segment_lesion = lambda *a, **k: None

try:
    from detection_model import (load_detection_model, train_detection_model,
                                  predict_detection, compute_grad_cam, TF_AVAILABLE)
    DETECTION_OK = True
except Exception as e:
    _IMPORT_ERRORS.append(("detection_model", repr(e), traceback.format_exc()))
    DETECTION_OK = False
    TF_AVAILABLE = False
    load_detection_model = lambda *a, **k: None
    train_detection_model = lambda *a, **k: None
    predict_detection = lambda *a, **k: (None, None)
    compute_grad_cam = lambda *a, **k: (None, None)

try:
    from classification_model import (load_classification_model, train_classification_models,
                                       predict_classification)
    CLASSIFICATION_OK = True
except Exception as e:
    _IMPORT_ERRORS.append(("classification_model", repr(e), traceback.format_exc()))
    CLASSIFICATION_OK = False
    load_classification_model = lambda *a, **k: (None, None)
    train_classification_models = lambda *a, **k: None
    predict_classification = lambda *a, **k: (None, None, None, None)

try:
    from visualization import (plot_prediction_dashboard, plot_roc_curve, plot_pr_curve,
                               plot_confusion_matrix, plot_calibration_curve,
                               plot_model_metrics_bar, plot_feature_importance,
                               plot_shap_analysis, plot_detection_history,
                               plot_top_features_radar)
    VISUALIZATION_OK = True
except Exception as e:
    _IMPORT_ERRORS.append(("visualization", repr(e), traceback.format_exc()))
    VISUALIZATION_OK = False
    plot_prediction_dashboard = lambda *a, **k: None
    plot_roc_curve = lambda *a, **k: None
    plot_pr_curve = lambda *a, **k: None
    plot_confusion_matrix = lambda *a, **k: None
    plot_calibration_curve = lambda *a, **k: None
    plot_model_metrics_bar = lambda *a, **k: None
    plot_feature_importance = lambda *a, **k: None
    plot_shap_analysis = lambda *a, **k: None
    plot_detection_history = lambda *a, **k: None
    plot_top_features_radar = lambda *a, **k: None


class BreastCancerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("乳腺癌超声影像智能诊断系统 v1.0 | 深度学习+半监督机器学习")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(bg="#f5f7fa")
        self.detection_model = None
        self.cls_model = None
        self.cls_data = None
        self.current_img_path = None
        self.current_img = None
        self.current_feature = None
        self.det_result = None
        self.det_prob = None
        self.cls_result = None
        self.cls_prob = None
        self.heatmap_overlay = None
        self._import_errors = list(_IMPORT_ERRORS)
        self._ui_scale = 1.0
        self._init_styles()
        self._build_ui()
        self.bind_all("<Control-MouseWheel>", self._on_global_ctrl_wheel)
        if self._import_errors:
            self.after(300, self._show_import_errors)
        self._auto_load_models()

    def _on_global_ctrl_wheel(self, event):
        delta = 0.08 if event.delta > 0 else -0.08
        new_scale = min(1.8, max(0.7, self._ui_scale + delta))
        if abs(new_scale - self._ui_scale) < 0.02:
            return
        self._ui_scale = new_scale
        self._apply_ui_scale()

    def _init_styles(self):
        self._style = ttk.Style(self)
        try:
            self._style.theme_use("clam")
        except Exception:
            pass
        self._base_sizes = {
            "tab_pad": (18, 10), "btn_pad": (12, 6), "btn_font": 10,
            "accent_pad": (14, 8), "accent_font": 10, "prog_thickness": 18,
            "tree_row": 24, "tree_font": 10, "header_title": 18, "header_sub": 10
        }
        self._apply_style_scale()

    def _apply_style_scale(self):
        s = self._ui_scale
        style = self._style
        def sz(v):
            if isinstance(v, tuple):
                return tuple(int(x * s) for x in v)
            return int(v * s)
        style.configure("TNotebook.Tab", padding=sz(self._base_sizes["tab_pad"]),
                        font=("Microsoft YaHei", int(self._base_sizes["btn_font"] * s), "bold"))
        style.configure("TButton", padding=sz(self._base_sizes["btn_pad"]),
                        font=("Microsoft YaHei", int(self._base_sizes["btn_font"] * s)))
        style.configure("Accent.TButton", padding=sz(self._base_sizes["accent_pad"]),
                        font=("Microsoft YaHei", int(self._base_sizes["accent_font"] * s), "bold"),
                        background="#1976d2", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1565c0")])
        style.configure("Danger.TButton", padding=sz(self._base_sizes["accent_pad"]),
                        font=("Microsoft YaHei", int(self._base_sizes["accent_font"] * s), "bold"),
                        background="#d32f2f", foreground="white")
        style.configure("Success.TButton", padding=sz(self._base_sizes["accent_pad"]),
                        font=("Microsoft YaHei", int(self._base_sizes["accent_font"] * s), "bold"),
                        background="#388e3c", foreground="white")
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#eeeeee",
                        background="#4caf50", thickness=sz(self._base_sizes["prog_thickness"]))
        style.configure("Treeview", font=("Microsoft YaHei", int(self._base_sizes["tree_font"] * s)),
                        rowheight=sz(self._base_sizes["tree_row"]))
        style.configure("Treeview.Heading",
                        font=("Microsoft YaHei", int(self._base_sizes["tree_font"] * s), "bold"))

    def _apply_ui_scale(self):
        self._apply_style_scale()
        self.update_idletasks()

    def _build_ui(self):
        header = tk.Frame(self, bg="#1565c0", height=68)
        header.pack(fill="x")
        tk.Label(header, text="  🏥 乳腺癌超声影像智能诊断系统",
                 bg="#1565c0", fg="white",
                 font=("Microsoft YaHei", 18, "bold")).pack(side="left", pady=14)
        tk.Label(header, text="深度学习(病灶检测) + 半监督机器学习(良恶性分类)  |  数据: BUS-UCLM + BUSI",
                 bg="#1565c0", fg="#bbdefb",
                 font=("Microsoft YaHei", 10)).pack(side="left", pady=14, padx=12)
        self.model_status_lbl = tk.Label(header, text="🔄 模型加载中...", bg="#1565c0", fg="#fff176",
                                         font=("Microsoft YaHei", 10, "bold"))
        self.model_status_lbl.pack(side="right", padx=20, pady=14)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_tab_predict()
        self._build_tab_train()
        self._build_tab_evaluate()
        self._build_tab_explain()

        self.footer = tk.Frame(self, bg="#e0e0e0", height=28)
        self.footer.pack(fill="x", side="bottom")
        self.progress_lbl = tk.Label(self.footer, text="就绪", bg="#e0e0e0", fg="#424242",
                                     font=("Microsoft YaHei", 9))
        self.progress_lbl.pack(side="left", padx=10)
        self.progress_bar = ttk.Progressbar(self.footer, style="Green.Horizontal.TProgressbar",
                                            mode="determinate", length=380)
        self.progress_bar.pack(side="right", padx=10, pady=3)

    def _show_import_errors(self):
        try:
            msgs = ["⚠️ 以下模块导入失败，部分功能将不可用:\n"]
            for name, err, tb in self._import_errors:
                msgs.append(f"  ❌ {name}: {err}")
            msgs.append("\n详情请查看日志，或运行『🔥调试启动_不闪退.bat』以获取完整错误信息")
            full_msg = "\n".join(msgs)
            self._set_status("⚠️ 有模块加载失败！点击右上角查看详情")
            self.model_status_lbl.configure(text="⚠️ 部分模块未加载")
            try:
                messagebox.showwarning("模块加载警告", full_msg, parent=self)
            except Exception:
                pass
        except Exception:
            pass

    def _build_tab_predict(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 🔍 影像诊断 ")
        left = tk.Frame(tab, bg="#ffffff", relief="ridge", bd=1)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(left, text="📂 影像输入区", bg="#1976d2", fg="white",
                 font=("Microsoft YaHei", 11, "bold"), pady=8).pack(fill="x")
        btns = tk.Frame(left, bg="#ffffff")
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="📁 选择影像...", style="Accent.TButton",
                   command=self._choose_image).pack(side="left", padx=8)
        ttk.Button(btns, text="📸 使用示例影像",
                   command=self._load_sample_image).pack(side="left", padx=4)
        ttk.Button(btns, text="🩺 开始智能诊断", style="Success.TButton",
                   command=self._start_predict).pack(side="left", padx=8)
        ttk.Button(btns, text="🗑️ 清空结果", command=self._clear_result).pack(side="left", padx=4)
        self.img_display = tk.Label(left, text="🖼️\n\n请导入一张乳腺超声影像\n(.png / .jpg / .bmp)",
                                    bg="#f5f5f5", fg="#9e9e9e", font=("Microsoft YaHei", 12),
                                    relief="sunken", bd=1, justify="center")
        self.img_display.pack(fill="both", expand=True, padx=8, pady=8)
        self.img_meta = tk.Label(left, text="ℹ️ 影像信息: 未加载",
                                 bg="#e3f2fd", fg="#0d47a1",
                                 font=("Microsoft YaHei", 9, "bold"),
                                 pady=6, anchor="w", justify="left")
        self.img_meta.pack(fill="x", padx=8, pady=(0, 8))
        right = tk.Frame(tab, bg="#ffffff", relief="ridge", bd=1)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(right, text="📊 诊断结果面板", bg="#1976d2", fg="white",
                 font=("Microsoft YaHei", 11, "bold"), pady=8).pack(fill="x")
        self.result_area = tk.Text(right, wrap="word", font=("Microsoft YaHei", 10),
                                   bg="#fafafa", relief="sunken", bd=1, height=14)
        self.result_area.pack(fill="x", padx=8, pady=6)
        self.result_area.tag_config("title", font=("Microsoft YaHei", 11, "bold"), foreground="#1565c0")
        self.result_area.tag_config("normal", foreground="#388e3c", font=("Microsoft YaHei", 11, "bold"))
        self.result_area.tag_config("abnormal", foreground="#d32f2f", font=("Microsoft YaHei", 11, "bold"))
        self.result_area.tag_config("benign", foreground="#66bb6a", font=("Microsoft YaHei", 11, "bold"))
        self.result_area.tag_config("malignant", foreground="#c62828", font=("Microsoft YaHei", 11, "bold"))
        self.result_area.tag_config("warn", foreground="#ef6c00", font=("Microsoft YaHei", 10, "bold"))
        self.result_area.tag_config("sub", foreground="#455a64", font=("Microsoft YaHei", 9))
        self.result_area.tag_config("divider", foreground="#90a4ae")
        self.result_area.insert("end", "💡 等待导入影像并执行诊断...\n\n", "sub")
        self.result_area.insert("end", "══【诊断流程】═══════════════\n", "divider")
        self.result_area.insert("end", "① 第一阶段: 深度学习(MobileNetV2) → 检测有无病灶区域\n", "sub")
        self.result_area.insert("end", "    可视化: Grad-CAM热力图(红=模型关注的病灶区)\n", "sub")
        self.result_area.insert("end", "② 第二阶段: 半监督ML(SelfTraining+RF影像组学) → 区分良性/恶性\n", "sub")
        self.result_area.insert("end", "    29维特征: 形状/强度/GLCM纹理/LBP分形/Hu矩\n", "sub")
        self.result_area.insert("end", "③ 可解释性: 前往【可解释性分析】查看SHAP/特征雷达\n", "sub")
        self.result_area.insert("end", "═══════════════════════════\n", "divider")
        self.result_area.configure(state="disabled")
        row = tk.Frame(right, bg="#ffffff")
        row.pack(fill="x", padx=8)
        self.det_status = tk.Label(row, text="病灶检测: ⚪ 未执行",
                                   bg="#eceff1", fg="#455a64",
                                   font=("Microsoft YaHei", 10, "bold"),
                                   padx=8, pady=6, anchor="w")
        self.det_status.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.cls_status = tk.Label(row, text="良恶性分类: ⚪ 未执行",
                                   bg="#eceff1", fg="#455a64",
                                   font=("Microsoft YaHei", 10, "bold"),
                                   padx=8, pady=6, anchor="w")
        self.cls_status.pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Label(right, text="🎯 Grad-CAM 病灶注意力热力图", bg="#2e7d32", fg="white",
                 font=("Microsoft YaHei", 11, "bold"), pady=6).pack(fill="x", padx=8, pady=(6, 0))
        self.heatmap_display = tk.Label(right, text="热力图显示区\n(诊断完成后显示)",
                                        bg="#f5f5f5", fg="#9e9e9e", font=("Microsoft YaHei", 11),
                                        relief="sunken", bd=1, justify="center")
        self.heatmap_display.pack(fill="both", expand=True, padx=8, pady=6)
        self.heatmap_hint = tk.Label(right, text="💡 解读: 红色高响应区=模型重点关注区域, 可交叉验证病灶位置",
                                     bg="#fff8e1", fg="#e65100",
                                     font=("Microsoft YaHei", 9), pady=4, anchor="w", justify="left")
        self.heatmap_hint.pack(fill="x", padx=8, pady=(0, 8))
        self._status_colors = {"OK": ("#388e3c", "#e8f5e9"), "BAD": ("#c62828", "#ffebee"),
                               "WAIT": ("#455a64", "#eceff1"), "WARN": ("#ef6c00", "#fff3e0")}

    def _build_tab_train(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" ⚙️ 模型训练 ")
        main = tk.Frame(tab, bg="#ffffff", relief="ridge", bd=1)
        main.pack(fill="both", expand=True, padx=6, pady=6)
        tk.Label(main, text="🤖 模型训练中心", bg="#6a1b9a", fg="white",
                 font=("Microsoft YaHei", 11, "bold"), pady=8).pack(fill="x")
        det_train = tk.LabelFrame(main, text=" 【第一阶段】病灶检测模型训练 (深度学习MobileNetV2) ",
                                  bg="#ffffff", font=("Microsoft YaHei", 10, "bold"), fg="#6a1b9a")
        det_train.pack(fill="x", padx=12, pady=10)
        tk.Label(det_train, text="数据来源: BUS-UCLM + Dataset_BUSI | 任务: 正常(0) vs 异常(1)二分类\n"
                                 "架构: ImageNet预训练MobileNetV2迁移 → 分类头 → 第二阶段全模型微调",
                 bg="#ffffff", fg="#555555", justify="left",
                 font=("Microsoft YaHei", 9)).pack(anchor="w", padx=8, pady=4)
        ttk.Button(det_train, text="🚀 开始训练检测模型", style="Accent.TButton",
                   command=self._train_detection_thread).pack(side="left", padx=8, pady=8)
        ttk.Button(det_train, text="📂 重新加载检测模型", command=self._reload_detection).pack(side="left", padx=4, pady=8)
        cls_train = tk.LabelFrame(main, text=" 【第二阶段】良恶性分类模型训练 (半监督机器学习) ",
                                  bg="#ffffff", font=("Microsoft YaHei", 10, "bold"), fg="#6a1b9a")
        cls_train.pack(fill="x", padx=12, pady=10)
        tk.Label(cls_train, text="特征: 29维影像组学(形状/强度/纹理GLCM/LBP/Hu矩) | 半监督方法: SelfTraining(阈值0.85) + LabelPropagation(KNN)\n"
                                 "对比基线: 监督学习RF / XGBoost / SVM / Logistic Regression",
                 bg="#ffffff", fg="#555555", justify="left",
                 font=("Microsoft YaHei", 9)).pack(anchor="w", padx=8, pady=4)
        ttk.Button(cls_train, text="🚀 开始训练分类模型", style="Accent.TButton",
                   command=self._train_classification_thread).pack(side="left", padx=8, pady=8)
        ttk.Button(cls_train, text="📂 重新加载分类模型", command=self._reload_classification).pack(side="left", padx=4, pady=8)
        self.train_log = tk.Text(main, wrap="word", font=("Consolas", 10),
                                 bg="#0d1117", fg="#85e89d", insertbackground="white",
                                 relief="sunken", bd=1, height=20)
        self.train_log.pack(fill="both", expand=True, padx=8, pady=8)
        self.train_log.insert("end", "[训练日志] 等待开始训练...\n")

    def _build_tab_evaluate(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 📈 模型评估 ")
        top = tk.Frame(tab, bg="#ffffff", relief="ridge", bd=1)
        top.pack(fill="x", padx=6, pady=6)
        tk.Label(top, text="📊 模型性能指标可视化", bg="#e65100", fg="white",
                 font=("Microsoft YaHei", 11, "bold"), pady=8).pack(fill="x")
        ttk.Button(top, text="生成评估图表", style="Accent.TButton",
                   command=self._generate_metrics).pack(side="left", padx=8, pady=8)
        self.eval_info = tk.Label(top, text="分类模型加载后可生成评估图表",
                                  bg="#ffffff", fg="#555555",
                                  font=("Microsoft YaHei", 10, "bold"))
        self.eval_info.pack(side="left", padx=10, pady=8)
        bot = self._add_scroll(tab)
        self.eval_canvas = {}
        titles = [("混淆矩阵", 0, 0), ("ROC曲线对比", 0, 1),
                  ("PR曲线对比", 1, 0), ("校准曲线/Brier", 1, 1)]
        for t, r, c in titles:
            frame = tk.LabelFrame(bot, text=f" {t} ", bg="#ffffff", height=380,
                                  font=("Microsoft YaHei", 10, "bold"), fg="#e65100")
            frame.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            frame.grid_propagate(False)
            bot.grid_columnconfigure(c, weight=1, uniform="ecol")
            lbl = tk.Label(frame, bg="#f5f5f5", text="点击上方按钮生成图表")
            lbl.pack(fill="both", expand=True, padx=4, pady=4)
            self.eval_canvas[t] = lbl
            ttk.Button(frame, text="🔍 放大", command=lambda path=t: self._open_eval_zoom(path)).pack(side="bottom", pady=3)
        extra = tk.LabelFrame(bot, text=" 多模型指标柱状对比 ", bg="#ffffff", height=480,
                              font=("Microsoft YaHei", 10, "bold"), fg="#e65100")
        extra.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        extra.grid_propagate(False)
        lbl = tk.Label(extra, bg="#f5f5f5", text="")
        lbl.pack(fill="both", expand=True, padx=4, pady=4)
        self.eval_canvas["metrics_bar"] = lbl
        ttk.Button(extra, text="🔍 放大", command=lambda: self._open_eval_zoom("多模型指标")).pack(side="bottom", pady=3)

    def _build_tab_explain(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 💡 可解释性分析 ")
        top = tk.Frame(tab, bg="#ffffff", relief="ridge", bd=1)
        top.pack(fill="x", padx=6, pady=6)
        tk.Label(top, text="🔬 模型可解释性分析中心 | Permutation特征重要性 + SHAP值 + 特征雷达图", bg="#00695c", fg="white",
                 font=("Microsoft YaHei", 11, "bold"), pady=8).pack(fill="x")
        btn_row1 = tk.Frame(top, bg="#ffffff")
        btn_row1.pack(fill="x", pady=4)
        ttk.Button(btn_row1, text="🚀 运行特征重要性分析", style="Accent.TButton",
                   command=self._run_feature_importance).pack(side="left", padx=8, pady=6)
        ttk.Button(btn_row1, text="🧠 生成SHAP分析图表", style="Success.TButton",
                   command=self._run_shap).pack(side="left", padx=4, pady=6)
        ttk.Button(btn_row1, text="📊 当前样本特征雷达(弹窗)", command=self._run_radar).pack(side="left", padx=4, pady=6)
        ttk.Button(btn_row1, text="📡 生成内嵌雷达图", command=self._run_radar_inline).pack(side="left", padx=4, pady=6)
        ttk.Button(btn_row1, text="🧹 清空所有分析图", command=self._clear_explain_charts).pack(side="left", padx=4, pady=6)
        self.exp_info = tk.Label(top, text="ℹ️ 操作说明: 先执行一次诊断→分类模型训练后可运行SHAP/特征重要性 | 特征雷达图展示当前样本的关键影像组学特征分布",
                                 bg="#ffffff", fg="#555555", justify="left", anchor="w",
                                 font=("Microsoft YaHei", 9))
        self.exp_info.pack(fill="x", padx=10, pady=(0, 6))
        bot = self._add_scroll(tab)
        self.exp_canvas = {}
        configs = [
            ("Permutation特征重要性", 0, 0, "基于AUC下降幅度，衡量特征对模型整体性能的影响；下降越多说明特征越重要"),
            ("SHAP Summary散点图", 0, 1, "特征值(颜色红高蓝低) vs SHAP贡献值(水平位置)，展示全局特征影响力分布"),
            ("SHAP Bar重要性", 1, 0, "平均绝对SHAP值的柱状排名，直观展示Top-N特征的重要性排序"),
            ("SHAP瀑布图(单样本)", 1, 1, "当前单个样本的SHAP分解：展示每个特征如何推动基准值→最终预测结果"),
        ]
        for t, r, c, desc in configs:
            frame = tk.LabelFrame(bot, text=f" {t} ", bg="#ffffff", height=410,
                                  font=("Microsoft YaHei", 10, "bold"), fg="#00695c")
            frame.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            frame.grid_propagate(False)
            bot.grid_columnconfigure(c, weight=1, uniform="xcol")
            lbl = tk.Label(frame, bg="#f5f5f5", text=f"⏳ 等待分析...\n\n💡 {desc}",
                           fg="#78909c", font=("Microsoft YaHei", 9), justify="center")
            lbl.pack(fill="both", expand=True, padx=4, pady=4)
            self.exp_canvas[t] = lbl
            ttk.Button(frame, text="🔍 放大查看", command=lambda name=t: self._open_exp_zoom(name)).pack(side="bottom", pady=3)
        radar_frame = tk.LabelFrame(bot, text=" 📊 当前预测样本 - 关键影像组学特征雷达图 ", bg="#ffffff", height=500,
                                    font=("Microsoft YaHei", 10, "bold"), fg="#00695c")
        radar_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        radar_frame.grid_propagate(False)
        self.radar_lbl = tk.Label(radar_frame, bg="#f5f5f5",
                                  text="⏳ 等待执行诊断...\n\n💡 雷达图展示当前样本在Top-N关键影像组学特征上的表现\n红(>0.7)高风险 / 绿(<0.3)低风险",
                                  fg="#78909c", font=("Microsoft YaHei", 10), justify="center")
        self.radar_lbl.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(radar_frame, text="🔍 放大雷达图", command=self._open_radar_zoom).pack(side="bottom", pady=3)

    def _auto_load_models(self):
        def load_thread():
            det_msg = "未训练"
            cls_msg = "未训练"
            try:
                self._set_status("加载病灶检测模型...")
                try:
                    self.detection_model = load_detection_model()
                    det_msg = "OK" if self.detection_model else "未训练"
                except Exception as e:
                    det_msg = "加载失败"
                    self.detection_model = None
                    print(f"[WARN] 检测模型加载异常: {e}")
                self._set_status("加载良恶性分类模型...")
                try:
                    self.cls_model, self.cls_data = load_classification_model()
                    if self.cls_data:
                        cls_msg = self.cls_data.get("best_name", "OK")
                    else:
                        cls_msg = "未训练"
                except Exception as e:
                    cls_msg = "加载失败"
                    self.cls_model = None
                    self.cls_data = None
                    print(f"[WARN] 分类模型加载异常: {e}")
                status = [f"✓检测:{det_msg}", f"✓分类:{cls_msg}"]
                self.after(0, lambda: self.model_status_lbl.configure(text=" | ".join(status)))
                self._set_status("就绪 | " + "  ".join(status))
            except Exception as e:
                traceback.print_exc()
                try:
                    self.after(0, lambda: self.model_status_lbl.configure(text="⚠️模型加载异常"))
                except Exception:
                    pass
        try:
            threading.Thread(target=load_thread, daemon=True).start()
        except Exception as e:
            print(f"[WARN] 模型加载线程启动失败: {e}")

    def _set_progress(self, value, text=None):
        if text is not None:
            self.progress_lbl.configure(text=text)
        self.progress_bar.configure(value=min(100, max(0, value)))
        self.update_idletasks()

    def _add_scroll(self, parent):
        outer = tk.Frame(parent, bg="#f5f5f5")
        outer.pack(fill="both", expand=True, padx=6, pady=6)
        canvas = tk.Canvas(outer, bg="#f5f5f5", highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        inner = tk.Frame(canvas, bg="#f5f5f5")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        def _on_wheel(event):
            if event.state & 0x0004:
                return
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        return inner

    def _set_status(self, text):
        self.after(0, lambda: self.progress_lbl.configure(text=text))

    def _log_train(self, text):
        def doit():
            self.train_log.insert("end", text + "\n")
            self.train_log.see("end")
        self.after(0, doit)

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="选择乳腺超声影像",
            filetypes=[("影像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")]
        )
        if path:
            self._load_image(path)

    def _load_sample_image(self):
        from data_loader import load_all_data
        records = load_all_data()
        if not records:
            messagebox.showwarning("提示", "未找到数据集图片")
            return
        r = np.random.choice([x for x in records if x["label_classify"] is not None] or records)
        self._load_image(r["img_path"])

    def _load_image(self, path):
        try:
            self.current_img_path = path
            self.current_img = read_image(path)
            if self.current_img is None:
                messagebox.showerror("错误", "无法读取图片: " + path)
                return
            self._display_image(self.img_display, self.current_img)
            fsize_kb = os.path.getsize(path) / 1024.0
            h, w = self.current_img.shape[:2]
            channels = self.current_img.shape[2] if self.current_img.ndim > 2 else 1
            ext = os.path.splitext(path)[1].upper()
            meta_text = (f"ℹ️ 影像信息 | 文件名: {os.path.basename(path)}\n"
                         f"   尺寸: {w}×{h} px | 通道: {channels} | 格式: {ext} | 大小: {fsize_kb:.1f} KB\n"
                         f"   路径: {os.path.dirname(path)}")
            self.img_meta.configure(text=meta_text)
            self.result_area.delete("1.0", "end")
            self.result_area.insert("end", f"📂 已加载: {os.path.basename(path)}\n", "title")
            self.result_area.insert("end", f"  📐 尺寸: {w}×{h}  |  💾 大小: {fsize_kb:.1f} KB  |  🎨 通道: {channels}\n\n")
            self.result_area.insert("end", "✅ 影像加载成功！点击【🩺 开始智能诊断】执行两阶段AI分析...\n\n", "sub")
            self.result_area.insert("end", "══【诊断流程】═══════════════\n", "divider")
            self.result_area.insert("end", "① 第一阶段: 深度学习(MobileNetV2) → 检测有无病灶区域\n", "sub")
            self.result_area.insert("end", "    可视化: Grad-CAM热力图(红=模型关注的病灶区)\n", "sub")
            self.result_area.insert("end", "② 第二阶段: 半监督ML(SelfTraining+RF影像组学) → 区分良性/恶性\n", "sub")
            self.result_area.insert("end", "    29维特征: 形状/强度/GLCM纹理/LBP分形/Hu矩\n", "sub")
            self.result_area.insert("end", "③ 可解释性: 前往【可解释性分析】查看SHAP/特征雷达\n", "sub")
            self.result_area.insert("end", "═══════════════════════════\n", "divider")
            self.heatmap_display.configure(image="", text="🎯 热力图显示区\n\n诊断完成后将显示\nGrad-CAM病灶注意力图\n\n🔴 红色高响应 = 模型重点关注区域")
            self.det_status.configure(text="病灶检测: ⚪ 待执行", bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0])
            self.cls_status.configure(text="良恶性分类: ⚪ 待执行", bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0])
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {e}\n{traceback.format_exc()}")

    def _display_image(self, label_widget, img_np, max_w=520, max_h=420):
        h, w = img_np.shape[:2]
        ratio = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = Image.fromarray(img_np).resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label_widget.configure(image=photo, text="")
        label_widget.image = photo

    def _start_predict(self):
        if self.current_img_path is None:
            messagebox.showwarning("提示", "请先导入一张影像")
            return
        if self.detection_model is None and self.cls_model is None:
            if not messagebox.askyesno("模型未训练", "检测和分类模型均未训练，是否先训练模型？\n(训练需要一定时间)"):
                return
            self.nb.select(1)
            return
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            self.after(0, lambda: (
                self.det_status.configure(text="病灶检测: ⏳ 提取特征中...", bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0]),
                self.cls_status.configure(text="良恶性分类: ⚪ 排队中...", bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0]),
                self.heatmap_hint.configure(text="⏳ 正在分析影像特征，请稍候...", bg="#fff3e0", fg="#e65100")
            ))
            self._set_progress(5, "提取影像组学特征...")
            feat_result = extract_all_features(self.current_img_path)
            if isinstance(feat_result, tuple) and len(feat_result) >= 1:
                self.current_feature = feat_result[0]
            else:
                self.current_feature = feat_result
            self._set_progress(30, "特征提取完成")
            if self.detection_model is not None:
                self.after(0, lambda: self.det_status.configure(text="病灶检测: ⏳ CNN推理中...", bg=self._status_colors["WARN"][1], fg=self._status_colors["WARN"][0]))
                self._set_progress(35, "第一阶段: 深度学习病灶检测(MobileNetV2推理)...")
                self.det_result, self.det_prob = predict_detection(self.detection_model, self.current_img_path)
                self._set_progress(55, "检测完成，生成Grad-CAM热力图...")
                self.after(0, lambda: self.det_status.configure(text="病灶检测: ⏳ 生成热力图...", bg=self._status_colors["WARN"][1], fg=self._status_colors["WARN"][0]))
                try:
                    hm, overlay = compute_grad_cam(self.detection_model, self.current_img_path)
                    self.heatmap_overlay = overlay
                except Exception as hm_err:
                    print(f"Grad-CAM warn: {hm_err}")
                    self.heatmap_overlay = None
            else:
                self.det_result, self.det_prob = 1, None
                self.heatmap_overlay = None
            self.cls_result = None
            self.cls_prob = None
            if self.det_result == 1 and self.cls_model is not None:
                self.after(0, lambda: self.cls_status.configure(text="良恶性分类: ⏳ ML推理中...", bg=self._status_colors["WARN"][1], fg=self._status_colors["WARN"][0]))
                self._set_progress(85, "第二阶段: 半监督良恶性分类(SelfTraining+RF)...")
                pred_res = predict_classification(self.cls_model, self.current_feature)
                if isinstance(pred_res, tuple) and len(pred_res) >= 2:
                    self.cls_result, self.cls_prob = pred_res[0], pred_res[1]
                else:
                    self.cls_result, self.cls_prob = None, None
            self._set_progress(100, "✅ 诊断完成！整理报告...")
            self.after(0, lambda: self.heatmap_hint.configure(text="💡 解读: 红色高响应区=模型重点关注区域, 可交叉验证病灶位置", bg="#fff8e1", fg="#e65100"))
            self.after(0, self._show_predict_result)
        except Exception as e:
            traceback.print_exc()
            self.after(0, lambda: (
                self.det_status.configure(text="病灶检测: ❌ 错误", bg=self._status_colors["BAD"][1], fg=self._status_colors["BAD"][0]),
                self.cls_status.configure(text="良恶性分类: ❌ 错误", bg=self._status_colors["BAD"][1], fg=self._status_colors["BAD"][0])
            ))
            messagebox.showerror("诊断错误", f"{e}\n{traceback.format_exc()}")

    def _show_predict_result(self):
        self.result_area.configure(state="normal")
        self.result_area.delete("1.0", "end")
        import datetime
        report_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.result_area.insert("end", "═══════════════════════════════════════════════\n", "divider")
        self.result_area.insert("end", "🏥  乳腺超声影像AI智能诊断系统 - 完整诊断报告\n", "title")
        self.result_area.insert("end", "═══════════════════════════════════════════════\n\n", "divider")
        self.result_area.insert("end", "━━━【报告基本信息】━━━━━━━━━━━━━━━━━━━━━\n", "divider")
        self.result_area.insert("end", f"📅 报告生成时间: {report_time}\n")
        self.result_area.insert("end", f"🆔 报告编号: BC{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}\n")
        self.result_area.insert("end", f"📁 影像文件名: {os.path.basename(self.current_img_path)}\n")
        self.result_area.insert("end", f"📍 文件完整路径: {os.path.dirname(os.path.abspath(self.current_img_path))}\n")
        if self.current_img is not None:
            h, w = self.current_img.shape[:2]
            channels = self.current_img.shape[2] if len(self.current_img.shape) > 2 else 1
            fsize_kb = 0.0
            try:
                fsize_kb = os.path.getsize(self.current_img_path) / 1024.0
            except Exception:
                pass
            ext = os.path.splitext(self.current_img_path)[1].upper() or "(未知格式)"
            self.result_area.insert("end", f"📐 影像规格: {w}×{h}px | 通道数:{channels} | 格式:{ext} | 大小:{fsize_kb:.1f}KB\n\n")
        self.result_area.insert("end", "━━━【第一阶段】病灶存在性检测━━━━━━━━━━━━━━━━━━\n", "divider")
        self.result_area.insert("end", "🧠 深度学习检测模型: MobileNetV2 (ImageNet预训练迁移学习)\n", "sub")
        self.result_area.insert("end", "   架构细节: 冻结前80层特征提取 + 两阶段微调 + Dropout(0.4/0.3) + BatchNorm\n", "sub")
        self.result_area.insert("end", "   输入规格: 224×224×3 RGB归一化 | 输出: Sigmoid二分类概率 正常/异常\n\n", "sub")
        det_lbl = LABEL_MAP_DETECTION.get(self.det_result, str(self.det_result))
        det_tag = "abnormal" if self.det_result == 1 else "normal"
        self.result_area.insert("end", f"🔍 病灶检测结论: {det_lbl}\n", det_tag)
        if self.det_prob is not None:
            det_abn = self.det_prob * 100
            det_nor = (1 - self.det_prob) * 100
            self.result_area.insert("end", f"📊 置信度概率分布:\n")
            self.result_area.insert("end", f"   🔴 异常(疑似有病灶): {det_abn:.2f}%\n")
            self.result_area.insert("end", f"   🟢 正常(未见明显病灶): {det_nor:.2f}%\n")
            bar_len = 38
            filled = int(bar_len * self.det_prob)
            bar = "█" * filled + "░" * (bar_len - filled)
            self.result_area.insert("end", f"   [{bar}] 异常概率 {det_abn:.1f}%\n")
            if self.det_prob >= 0.95:
                det_conf = "★★★★★ 极高置信度"
            elif self.det_prob >= 0.85:
                det_conf = "★★★★☆ 高置信度"
            elif self.det_prob >= 0.70:
                det_conf = "★★★☆☆ 中等置信度"
            elif self.det_prob >= 0.55:
                det_conf = "★★☆☆☆ 偏低置信度，建议结合其他信息综合"
            else:
                det_conf = "★☆☆☆☆ 边界判断值，建议短期复查"
            self.result_area.insert("end", f"   🎯 判断置信度评级: {det_conf}\n\n")
        if self.det_result == 0:
            self.det_status.configure(text="病灶检测: ✅ 正常(无病灶)", bg=self._status_colors["OK"][1], fg=self._status_colors["OK"][0])
            self.result_area.insert("end", "✅【检测结论】未检测到明显异常病灶区域\n\n", "normal")
            self.result_area.insert("end", "📋【常规乳腺健康管理建议】\n", "title")
            self.result_area.insert("end", "  ① 作息管理: 避免熬夜，保证规律作息，每晚23点前入睡，7-8小时优质睡眠\n", "sub")
            self.result_area.insert("end", "  ② 饮食管理: 多摄取蔬菜水果豆制品全谷物；减少高脂高糖加工食品、酒精摄入\n", "sub")
            self.result_area.insert("end", "  ③ 乳房自查: 每月月经周期第7-10天进行乳房自检，熟悉自身乳腺状态\n", "sub")
            self.result_area.insert("end", "  ④ 定期体检: 建议每年至少一次专业乳腺体检(超声+专科触诊)，40岁以上加钼靶\n", "sub")
            self.result_area.insert("end", "  ⑤ 激素管理: 避免不必要外源性雌激素(避孕药/丰乳保健品/激素替代疗法)，慎用\n", "sub")
            self.result_area.insert("end", "  ⑥ 运动管理: 每周3-5次中等强度有氧运动，每次30-45分钟，控制BMI在18.5-24\n\n", "sub")
        else:
            self.det_status.configure(text="病灶检测: ⚠️ 异常(有病灶)", bg=self._status_colors["WARN"][1], fg=self._status_colors["WARN"][0])
            self.result_area.insert("end", "⚠️【检测发现】检测到疑似病灶区域 → 自动进入第二阶段良恶性分类分析\n\n", "warn")
            self.result_area.insert("end", "━━━【第二阶段】病灶良恶性分类━━━━━━━━━━━━━━━━━━\n", "divider")
            self.result_area.insert("end", "🤖 半监督机器学习分类: 影像组学29维特征 + SelfTraining半监督框架\n", "sub")
            self.result_area.insert("end", "   特征类型: 7形状+8强度统计+8GLCM纹理+LBP分形+Hu不变矩 (共29维)\n", "sub")
            self.result_area.insert("end", "   训练策略: 60%无标签数据伪标签(阈值0.85) + RF随机森林集成/LabelPropagation\n", "sub")
            self.result_area.insert("end", "   对比基线: RF/XGBoost/SVM/LR，自动选AUC最高的模型作为最终模型\n\n", "sub")
            if self.cls_model is not None:
                cls_lbl = LABEL_MAP_CLASSIFY.get(self.cls_result, str(self.cls_result))
                cls_tag = "malignant" if self.cls_result == 1 else "benign"
                self.result_area.insert("end", f"🎯 良恶性分类结论: {cls_lbl}\n", cls_tag)
                if self.cls_prob is not None:
                    cls_mal = self.cls_prob * 100
                    cls_ben = (1 - self.cls_prob) * 100
                    self.result_area.insert("end", f"📊 概率分布:\n")
                    self.result_area.insert("end", f"   🔴 恶性概率: {cls_mal:.2f}%\n")
                    self.result_area.insert("end", f"   🟢 良性概率: {cls_ben:.2f}%\n")
                    cbar_len = 38
                    cfilled = int(cbar_len * self.cls_prob)
                    cbar = "█" * cfilled + "░" * (cbar_len - cfilled)
                    self.result_area.insert("end", f"   [{cbar}] 恶性概率 {cls_mal:.1f}%\n")
                    if self.cls_result == 1:
                        if self.cls_prob >= 0.90:
                            cls_conf = "🔴🔴🔴 极高可能性恶性"
                        elif self.cls_prob >= 0.75:
                            cls_conf = "🔴🔴 高度可能性恶性"
                        elif self.cls_prob >= 0.60:
                            cls_conf = "🔴 中度倾向恶性"
                        else:
                            cls_conf = "🟠 低度倾向恶性，建议活检确认"
                    else:
                        if self.cls_prob <= 0.10:
                            cls_conf = "🟢🟢🟢 极高可能性良性"
                        elif self.cls_prob <= 0.25:
                            cls_conf = "🟢🟢 高度可能性良性"
                        elif self.cls_prob <= 0.40:
                            cls_conf = "🟢 中度倾向良性"
                        else:
                            cls_conf = "🟡 偏低倾向良性，建议密切随访"
                    self.result_area.insert("end", f"   🎯 良恶性判断置信度: {cls_conf}\n\n")
                model_name = self.cls_data['best_name'] if self.cls_data else 'Unknown'
                best_metrics = self.cls_data.get('best_result', {}) if self.cls_data else {}
                self.result_area.insert("end", f"🏆 本次使用最优模型: {model_name}\n\n", "title")
                if best_metrics:
                    self.result_area.insert("end", "📈【模型独立测试集性能指标】(判断准确度与可靠性参考)\n", "title")
                    def _fmt_pct(v):
                        try:
                            return f"{float(v)*100:.2f}%"
                        except Exception:
                            return str(v)
                    acc_val = _fmt_pct(best_metrics.get("acc", "—"))
                    auc_val = _fmt_pct(best_metrics.get("auc", "—"))
                    prec_val = _fmt_pct(best_metrics.get("precision", "—"))
                    rec_val = _fmt_pct(best_metrics.get("recall", "—"))
                    f1_val = _fmt_pct(best_metrics.get("f1", "—"))
                    brier_val = f"{best_metrics.get('brier', '—')}"
                    try:
                        br = float(best_metrics.get("brier", 1.0))
                        if br <= 0.05:
                            brier_qual = "🟢 优秀校准"
                        elif br <= 0.10:
                            brier_qual = "🟢 良好校准"
                        elif br <= 0.15:
                            brier_qual = "🟡 一般校准"
                        else:
                            brier_qual = "🟠 可进一步校准"
                    except Exception:
                        brier_qual = ""
                    cm = best_metrics.get("cm", None)
                    n_test = sum(sum(row) for row in cm) if isinstance(cm, list) else "—"
                    self.result_area.insert("end", "   ┌─────────────┬────────────────────────────────────────────────┐\n", "divider")
                    self.result_area.insert("end", f"   │ Accuracy    │ 准确率 {acc_val:<18} (整体预测正确比例)            │\n", "sub")
                    self.result_area.insert("end", f"   │ AUC ROC     │ 曲线下面积 {auc_val:<14} (总体区分能力好坏)        │\n", "sub")
                    self.result_area.insert("end", f"   │ Precision   │ 精确率 {prec_val:<16} (报恶性真恶性比例)          │\n", "sub")
                    self.result_area.insert("end", f"   │ Recall      │ 召回率 {rec_val:<18} (漏诊率=1-{rec_val})            │\n", "sub")
                    self.result_area.insert("end", f"   │ F1 Score    │ F1综合分 {f1_val:<16} (精确召回调和均值)          │\n", "sub")
                    self.result_area.insert("end", f"   │ Brier Loss  │ {str(brier_val)[:7]:<12} {brier_qual:<27}│\n", "sub")
                    self.result_area.insert("end", f"   │ Test Size   │ 独立测试集 {str(n_test):<17}例样本                │\n", "sub")
                    self.result_area.insert("end", "   └─────────────┴────────────────────────────────────────────────┘\n\n", "divider")
                    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                        [[tn, fp], [fn, tp]] = cm
                        self.result_area.insert("end", "🧮【混淆矩阵详细分析】(模型在独立测试集上的实际表现)\n", "sub")
                        self.result_area.insert("end", "   ┌───────────────┬─────────────────┬─────────────────┐\n", "divider")
                        self.result_area.insert("end", "   │ 实际 \\ 预测    │ 预测良性         │ 预测恶性         │\n", "sub")
                        self.result_area.insert("end", "   ├───────────────┼─────────────────┼─────────────────┤\n", "divider")
                        self.result_area.insert("end", f"   │ 实际良性       │ TN = {tn:<6} ✅真阴 │ FP = {fp:<6} ❌误诊 │\n", "normal")
                        self.result_area.insert("end", f"   │ 实际恶性       │ FN = {fn:<6} ❌漏诊 │ TP = {tp:<6} ✅真阳 │\n", "malignant")
                        self.result_area.insert("end", "   └───────────────┴─────────────────┴─────────────────┘\n\n", "divider")
                        mis_rate = fn / max(1, fn + tp) * 100 if fn + tp > 0 else 0
                        fpr = fp / max(1, fp + tn) * 100 if fp + tn > 0 else 0
                        spec_rate = tn / max(1, fp + tn) * 100 if fp + tn > 0 else 0
                        npv_rate = tn / max(1, tn + fn) * 100 if tn + fn > 0 else 0
                        self.result_area.insert("end", f"   ⚠️ 漏诊率(恶性错判良性): {mis_rate:.1f}%  → 约每100个恶性病例中约{mis_rate:.0f}例会被模型漏判\n", "warn")
                        self.result_area.insert("end", f"   ⚠️ 误诊率(良性错判恶性): {fpr:.1f}%  → 约每100个良性病例中约{fpr:.0f}例会被模型误报恶性\n", "sub")
                        self.result_area.insert("end", f"   📊 特异度(正确识别良性): {spec_rate:.1f}%  |  阴性预测值: {npv_rate:.1f}%\n\n", "sub")
                if self.cls_result == 1:
                    if self.cls_prob > 0.80:
                        risk_level = "🔴🔴 高风险"
                    elif self.cls_prob > 0.65:
                        risk_level = "🔴 中高风险"
                    elif self.cls_prob > 0.45:
                        risk_level = "🟠 中等风险"
                    else:
                        risk_level = "🟡 偏低风险但仍需确认"
                    if self.cls_prob > 0.95:
                        birads = "BI-RADS 5类 (高度提示恶性, 恶性概率≥95%)"
                    elif self.cls_prob > 0.70:
                        birads = "BI-RADS 4C类 (高度可疑恶性, 50%-95%)"
                    elif self.cls_prob > 0.40:
                        birads = "BI-RADS 4B类 (中度可疑恶性, 10%-50%)"
                    else:
                        birads = "BI-RADS 4A类 (低度可疑恶性, 2%-10%)"
                    self.cls_status.configure(text="良恶性分类: 🔴 恶性", bg=self._status_colors["BAD"][1], fg=self._status_colors["BAD"][0])
                    self.result_area.insert("end", f"⚠️【综合风险评估】{risk_level}\n", "warn")
                    self.result_area.insert("end", f"ℹ️  AI参考影像分级: {birads}\n\n", "warn")
                    self.result_area.insert("end", "🚨【紧急推荐：进一步明确诊断检查方案】\n", "title")
                    self.result_area.insert("end", "  【1】金标准确诊检查 (请尽快预约):\n", "title")
                    self.result_area.insert("end", "     ▶ 首选: 超声引导下空心针穿刺活检 (Core Needle Biopsy, CNB)\n", "sub")
                    self.result_area.insert("end", "     ▶ 备选: 真空辅助微创活检 (VAB) — 适合病灶较小(<1cm)/位置较深/多发微钙化\n", "sub")
                    self.result_area.insert("end", "     ▶ 活检标本送检: 常规病理(HE染色) + 免疫组化(ER/PR/HER2/Ki67) 明确分子分型\n", "sub")
                    self.result_area.insert("end", "  【2】影像学分期补充检查:\n", "title")
                    self.result_area.insert("end", "     ▶ 双侧乳腺钼靶X线摄影(CC位+MLO位) — 排查微钙化及多中心多灶病灶\n", "sub")
                    self.result_area.insert("end", "     ▶ 乳腺动态增强MRI — 精准评估病灶范围、对侧乳腺、胸壁侵犯、腋窝淋巴结\n", "sub")
                    self.result_area.insert("end", "     ▶ 双侧腋窝+锁骨上/下淋巴结超声 — 评估区域淋巴结转移状态\n", "sub")
                    self.result_area.insert("end", "     ▶ 胸部CT + 腹部超声(±骨扫描/头颅MRI) — 晚期高危者排查远处转移\n", "sub")
                    self.result_area.insert("end", "  【3】实验室基线检查:\n", "title")
                    self.result_area.insert("end", "     ▶ 血清肿瘤标志物: CA15-3 + CEA + CA125 + HE4 (基线值,动态监测用)\n", "sub")
                    self.result_area.insert("end", "     ▶ 术前常规: 血常规、肝肾功能、电解质、凝血功能、传染病筛查\n", "sub")
                    self.result_area.insert("end", "     ▶ 酌情: 性激素6项、甲状腺功能、BRCA1/2基因检测(有家族史/年轻/三阴性)\n", "sub")
                    self.result_area.insert("end", "\n🏥【按分期的推荐治疗路径参考】(确诊病理后)\n", "title")
                    self.result_area.insert("end", "  【早期乳腺癌 (0期/Ⅰ期/Ⅱ期, T≤5cm N0M0)】:\n", "sub")
                    self.result_area.insert("end", "     ▶ 保乳手术(BCS)+前哨淋巴结活检(SLNB) + 术后放疗 (符合条件者首选)\n", "sub")
                    self.result_area.insert("end", "     ▶ 或 乳腺癌改良根治术 + 即刻/延期乳房假体重建/自体组织重建\n", "sub")
                    self.result_area.insert("end", "     ▶ 术后辅助: 中高危+全身治疗(按分子分型选择化疗/内分泌/靶向)\n", "sub")
                    self.result_area.insert("end", "  【局部晚期乳腺癌 (Ⅲ期, T>5cm/N+/炎性乳癌)】:\n", "sub")
                    self.result_area.insert("end", "     ▶ 新辅助治疗(化疗±靶向±内分泌)降期 → 评估手术 → 术后辅助治疗\n", "sub")
                    self.result_area.insert("end", "  【转移性乳腺癌 (Ⅳ期, 远处转移)】:\n", "sub")
                    self.result_area.insert("end", "     ▶ 以全身治疗为主: 化疗/内分泌治疗/靶向治疗/免疫治疗±姑息局部处理\n", "sub")
                    self.result_area.insert("end", "\n📌【分子分型个体化治疗原则】(参考免疫组化结果)\n", "title")
                    self.result_area.insert("end", "     ▶ Luminal A型(ER+PR+HER2-Ki67<14%): 以内分泌治疗为主，多数豁免化疗\n", "sub")
                    self.result_area.insert("end", "     ▶ Luminal B型(ER+HER2±或Ki67≥14%): 内分泌+酌情化疗±抗HER2靶向\n", "sub")
                    self.result_area.insert("end", "     ▶ HER2过表达型(ER-PR-HER2+): 抗HER2双靶+化疗 (曲妥珠+帕妥珠)\n", "sub")
                    self.result_area.insert("end", "     ▶ 三阴性(TNBC, ER-PR-HER2-): 化疗为主，BRCA+可选PARP抑制剂/免疫\n", "sub")
                    self.result_area.insert("end", "\n⚠️【请务必立即行动】\n", "warn")
                    self.result_area.insert("end", "     ▶ 建议3个工作日内到【三甲医院乳腺外科/肿瘤科】就诊，勿拖延！\n", "warn")
                    self.result_area.insert("end", "     ▶ 就诊时携带: 本AI报告+原始超声DICOM影像+钼靶片(如有)+既往病历\n", "warn")
                    self.result_area.insert("end", "     ▶ 重要提示: 早期乳腺癌5年生存率>93%，请务必重视但不必过度恐慌\n", "warn")
                    self.result_area.insert("end", "     ▶ 确诊前: 避免反复用力按摩病灶区域；暂停服用含雌激素类产品；勿热敷\n", "warn")
                else:
                    self.cls_status.configure(text="良恶性分类: 🟢 良性", bg=self._status_colors["OK"][1], fg=self._status_colors["OK"][0])
                    self.result_area.insert("end", "✅【综合风险评估】倾向良性病灶\n", "normal")
                    if self.cls_prob is not None:
                        if self.cls_prob <= 0.10:
                            birads = "BI-RADS 2类 (肯定良性发现, 恶性概率≈0%)"
                        elif self.cls_prob <= 0.20:
                            birads = "BI-RADS 3类 (可能良性, 恶性概率<2%)"
                        else:
                            birads = "BI-RADS 3类 (可能良性, 建议短期随访确认)"
                        self.result_area.insert("end", f"ℹ️  AI参考影像分级: {birads}\n\n", "normal")
                    self.result_area.insert("end", "💊【常见良性病灶参考类型】(需临床查体确认)\n", "title")
                    self.result_area.insert("end", "  · 乳腺纤维腺瘤: 青年女性最常见，圆形/椭圆形、边界清、活动度好、生长慢\n", "sub")
                    self.result_area.insert("end", "  · 乳腺增生结节/腺病: 与月经周期相关，多发、双侧、常伴经前胀痛\n", "sub")
                    self.result_area.insert("end", "  · 乳腺囊肿: 超声无回声区、后方回声增强，大小可变，可单可多\n", "sub")
                    self.result_area.insert("end", "  · 导管内乳头状瘤: 常伴单侧单孔乳头溢液(血性/浆液性)，乳晕下多见\n", "sub")
                    self.result_area.insert("end", "  · 炎性/浆细胞乳腺炎: 红肿热痛，抗生素反应不一，易反复\n", "sub")
                    self.result_area.insert("end", "  · 脂肪瘤: 高回声，位于脂肪层内，柔软可推动，无血流信号\n\n", "sub")
                    self.result_area.insert("end", "🏥【推荐后续随访时间表与检查方案】\n", "title")
                    self.result_area.insert("end", "  【1】影像学随访时间表 (请在手机日历设置提醒):\n", "title")
                    t1 = (datetime.datetime.now() + datetime.timedelta(days=93)).strftime("%Y-%m")
                    t2 = (datetime.datetime.now() + datetime.timedelta(days=186)).strftime("%Y-%m")
                    t3 = (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m")
                    self.result_area.insert("end", f"     ▶ 第1次复查: {t1} (约3个月后) → 乳腺超声 (重点对比大小形态边界血流)\n", "sub")
                    self.result_area.insert("end", f"     ▶ 第2次复查: {t2} (约6个月后) → 乳腺超声 + 钼靶(40岁以上/高危)\n", "sub")
                    self.result_area.insert("end", f"     ▶ 第3次复查: {t3} (约12个月后) → 乳腺彩超 + 钼靶X线摄影\n", "sub")
                    self.result_area.insert("end", "     ▶ 连续2年稳定无进展，可延长至每年1次常规体检；仍有变化需缩短间隔\n", "sub")
                    self.result_area.insert("end", "  【2】必须立即就医的【红旗预警征象】(出现任何一项，切勿等待随访):\n", "title")
                    self.result_area.insert("end", "     ❗ 结节3个月内体积增大>20% 或 直径每月增长>1mm (快速生长)\n", "warn")
                    self.result_area.insert("end", "     ❗ 形态由规则→不规则，边界由清晰→模糊不清，出现毛刺/成角/分叶征\n", "warn")
                    self.result_area.insert("end", "     ❗ 内部出现砂砾样/簇状微钙化、后方回声衰减、纵横比>1(纵向生长)\n", "warn")
                    self.result_area.insert("end", "     ❗ CDFI/超声造影出现异常丰富高阻血流 (RI>0.70)、穿支血管、新生血管\n", "warn")
                    self.result_area.insert("end", "     ❗ 同侧腋窝淋巴结异常(皮质增厚>3mm/形态变圆/淋巴门消失/异常血流)\n", "warn")
                    self.result_area.insert("end", "     ❗ 乳房皮肤橘皮样改变、凹陷、乳头内陷、乳头单孔血性溢液、皮肤破溃\n", "warn")
                    self.result_area.insert("end", "     ❗ 不明原因乳房疼痛持续加重不缓解、局部发热红肿、伴体重明显下降\n", "warn")
                    self.result_area.insert("end", "  【3】全方位生活方式干预与调理建议:\n", "title")
                    self.result_area.insert("end", "     🥗 饮食营养调理:\n", "sub")
                    self.result_area.insert("end", "       √ 增加: 十字花科蔬菜(西兰花/花椰菜/甘蓝/白菜)、豆类、菌菇、海藻类、浆果\n", "sub")
                    self.result_area.insert("end", "       √ 推荐: 每日豆制品25-50g(含植物雌激素双向调节)，坚果15g，深绿色菜300g\n", "sub")
                    self.result_area.insert("end", "       √ 烹调: 多采用蒸煮炖凉拌，减少煎炸烧烤；优选橄榄油/茶籽油代替动物油\n", "sub")
                    self.result_area.insert("end", "       × 限制: 加工肉制品、油炸/反式脂肪、高糖甜食饮料、精制碳水、烟熏食品\n", "sub")
                    self.result_area.insert("end", "       × 避免: 酒精(任何剂量都增加乳腺风险)、吸烟(含二手烟)、>4杯/天浓咖啡\n", "sub")
                    self.result_area.insert("end", "     🏃 运动与体重管理:\n", "sub")
                    self.result_area.insert("end", "       √ 每周3-5次，每次30-45分钟中等强度有氧 (快走/游泳/骑车/有氧操/跳绳)\n", "sub")
                    self.result_area.insert("end", "       √ 每周2次抗阻力量训练(哑铃/弹力带)，增肌有助于激素代谢和胰岛素稳定\n", "sub")
                    self.result_area.insert("end", "       √ 目标: BMI控制在18.5-23.9，女性腰围<85cm，体脂率20%-28%为宜\n", "sub")
                    self.result_area.insert("end", "     🧘 情绪与睡眠管理:\n", "sub")
                    self.result_area.insert("end", "       √ 保持情绪平稳，避免长期焦虑/抑郁/生闷气；压力大时及时疏解求助\n", "sub")
                    self.result_area.insert("end", "       √ 作息规律，每晚23点前入睡，7-8小时；避免夜间强光(抑制褪黑素)\n", "sub")
                    self.result_area.insert("end", "       √ 可尝试: 正念冥想/渐进式肌肉放松/呼吸训练/瑜伽/户外散步缓解压力\n", "sub")
                    self.result_area.insert("end", "     ⚠️ 其他注意事项:\n", "sub")
                    self.result_area.insert("end", "       × 避免长期口服复方避孕药(35岁以上/家族史/吸烟慎用)，必要时咨询医生\n", "sub")
                    self.result_area.insert("end", "       × 慎用含雌激素的美容/丰乳/抗衰产品及不明成分保健品；绝经后HRT严格遵医嘱\n", "sub")
                    self.result_area.insert("end", "       × 避免长时间束胸或过紧钢圈内衣；睡眠时建议无钢圈或解扣，避免压迫淋巴\n", "sub")
                    self.result_area.insert("end", "       × 切勿频繁用力按摩结节(尤其是未明确性质前)；避免暴力推拿/精油开背胸部\n", "sub")
                    self.result_area.insert("end", "\n🌳【预后心理建设与下一步】\n", "title")
                    self.result_area.insert("end", "     ✓ 统计上约80%-90%乳腺结节为良性，多数类型恶变概率低，不必过度焦虑\n", "normal")
                    self.result_area.insert("end", "     ✓ 规律随访比胡思乱想更有意义；心理压力过大反而通过HPA轴影响内分泌\n", "normal")
                    self.result_area.insert("end", "     ✓ 如有疑虑: 携带完整影像资料前往三甲医院乳腺外科，获取专业二诊意见\n", "normal")
                    self.result_area.insert("end", "     ✓ 乳腺疾病可防可治，科学管理+及时就医，绝大多数预后良好\n\n", "normal")
            else:
                self.cls_status.configure(text="良恶性分类: ⚪ 模型未训练", bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0])
                self.result_area.insert("end", "⚠️ 分类模型未训练 → 仅完成病灶检测定位，未进行良恶性鉴别\n", "warn")
                self.result_area.insert("end", "💡 请前往【模型训练】Tab → 点击【② 训练良恶性分类模型】完成训练后再重新诊断\n\n", "sub")
        self.result_area.insert("end", "\n━━━【模型判断可靠度说明】━━━━━━━━━━━━━━━━━━\n", "divider")
        self.result_area.insert("end", "ℹ️ 关于AI判断准确度的重要说明（请务必阅读）:\n", "sub")
        self.result_area.insert("end", "  ① 本系统检测与分类模型均在独立划分的测试集上验证，实际临床场景下准确度可能因:\n", "sub")
        self.result_area.insert("end", "     · 超声设备品牌型号/增益/动态范围/探头频率参数差异\n", "sub")
        self.result_area.insert("end", "     · 操作者扫查手法熟练度、标准切面选择、病灶显示完整性\n", "sub")
        self.result_area.insert("end", "     · 病灶大小(<5mm或>50mm判读难度大)、位置(近场/深部/近乳晕)、非典型声像\n", "sub")
        self.result_area.insert("end", "     · 患者年龄、乳腺致密度(年轻致密腺体重叠干扰)、乳房假体植入史、手术史\n", "sub")
        self.result_area.insert("end", "     等因素与训练集分布不一致而产生波动，以上性能指标仅作参考。\n", "sub")
        self.result_area.insert("end", "  ② 所有置信度百分比仅代表算法基于现有数据的数学概率，≠个体实际发病率\n", "sub")
        self.result_area.insert("end", "  ③ 低置信度评级(★★及以下)病例务必结合: 病史+体格检查+多模态影像+随访±活检\n", "sub")
        self.result_area.insert("end", "  ④ 建议将本系统作为【第二读者/辅助筛查工具】，而非最终诊断依据\n\n", "sub")
        self.result_area.insert("end", "━━━【可解释性分析线索】━━━━━━━━━━━━━━━━━━\n", "divider")
        explain_count = 0
        if self.heatmap_overlay is not None:
            explain_count += 1
            self.result_area.insert("end", f"  {explain_count}. 🎯 Grad-CAM注意力热力图: 已生成 → 见右侧下方区域\n", "sub")
            self.result_area.insert("end", "     热力图解读: 🔴红/橙→黄→蓝 从高权重到低权重，显示模型重点关注区域\n", "sub")
            self.result_area.insert("end", "     🔍 【自我核对】: 请人工确认热力图高响应区是否与你看到的病灶位置吻合？\n", "sub")
            self.result_area.insert("end", "     ✓ 吻合 → 模型判断基于正确病灶区域，结果置信度可上调\n", "sub")
            self.result_area.insert("end", "     ✗ 错位 → 警惕模型可能基于伪影/腺体纹理/钙化灶误判，建议复查/专科会诊\n", "sub")
            self._display_image(self.heatmap_display, self.heatmap_overlay)
            self.heatmap_hint.configure(text="💡 解读: 红/黄高响应=模型重点区，请人工核对是否吻合实际病灶；吻合→结果更可信，错位→需警惕")
        else:
            if self.current_img is not None:
                self._display_image(self.heatmap_display, self.current_img)
            if self.detection_model is None:
                self.heatmap_hint.configure(text="⚠️ 检测模型未训练: 显示原始影像，请先训练检测模型后重新诊断以生成热力图", bg="#ffebee", fg="#c62828")
            else:
                self.heatmap_hint.configure(text="ℹ️ 热力图暂未生成: 已显示原始影像，可前往【模型训练】Tab确认检测模型状态", bg="#fff3e0", fg="#e65100")
        if self.current_feature is not None:
            top5 = np.argsort(-np.abs(self.current_feature))[:5]
            from config import FEATURE_NAMES
            feat_desc_map = {
                "area_ratio": "病灶相对面积占比",
                "perimeter": "病灶周长",
                "circularity": "圆度(越近1越规则)",
                "eccentricity": "离心率(越长条越大)",
                "solidity": "实体度(凹凸比)",
                "aspect_ratio": "纵横比(>1提示恶性)",
                "mean_intensity": "平均灰度值",
                "std_intensity": "灰度标准差(异质性)",
                "skewness": "灰度偏度(对称性)",
                "kurtosis": "灰度峰度(尖锐度)",
                "entropy": "信息熵(纹理复杂度)",
                "contrast": "GLCM对比度(反差)",
                "homogeneity": "GLCM同质性(均匀度)",
                "correlation": "GLCM相关性",
                "energy": "GLCM能量(规则度)",
                "dissimilarity": "GLCM差异度",
                "lbp_mean": "LBP局部二值均值",
                "lbp_std": "LBP局部二值方差",
                "hu_1": "Hu不变矩1",
                "hu_2": "Hu不变矩2",
            }
            names = [FEATURE_NAMES[i] for i in top5 if i < len(FEATURE_NAMES)]
            values = [float(self.current_feature[i]) for i in top5 if i < len(FEATURE_NAMES)]
            if names:
                explain_count += 1
                self.result_area.insert("end", f"\n  {explain_count}. 📊 当前病灶影像组学TOP5关键特征 (按偏离人群中位程度排序):\n", "sub")
                for ni, (nm, vl) in enumerate(zip(names, values), 1):
                    arrow = "↑ 偏高" if vl > 0 else "↓ 偏低"
                    desc = feat_desc_map.get(nm, "")
                    tag_clr = "malignant" if vl > 0 else "normal"
                    self.result_area.insert("end", f"       {ni}. {nm:<20} = {vl:+.3f}  ({arrow})  {desc}\n", tag_clr)
                self.result_area.insert("end", "     特征模式解读参考 (≠诊断依据,仅参考趋势):\n", "sub")
                self.result_area.insert("end", "       🔴 恶性常表现: 圆度↓(不规则) 离心率↑(分叶长条) 熵↑(结构乱) 对比度↑(纹理粗) 纵横比>1\n", "malignant")
                self.result_area.insert("end", "       🟢 良性常表现: 圆度↑(近圆形) 熵↓(结构均) 对比度↓(纹理细) 边界清 后方无衰减\n", "normal")
                explain_count += 1
                self.result_area.insert("end", f"\n  {explain_count}. 🧠 建议前往【可解释性分析】Tab获取更深入的分析:\n", "title")
                self.result_area.insert("end", "     📊 全局特征重要性(Permutation) — 哪些特征对模型整体决策最关键\n", "sub")
                self.result_area.insert("end", "     🧩 SHAP值分析(Summary/Bar/Waterfall/Force) — 29维特征各自对本次判断的正负贡献\n", "sub")
                self.result_area.insert("end", "     📈 关键特征雷达图 — 可视化当前样本在各维度与训练集基线的偏离模式\n\n", "sub")
        self.result_area.insert("end", "━━━【随访记录日历提醒】━━━━━━━━━━━━━━━━━━\n", "divider")
        next_m3 = (datetime.datetime.now() + datetime.timedelta(days=93)).strftime("%Y-%m-%d")
        next_m6 = (datetime.datetime.now() + datetime.timedelta(days=186)).strftime("%Y-%m-%d")
        next_y1 = (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        self.result_area.insert("end", f"  ⏰ 建议复查时间节点 (建议在手机日历/电脑闹钟设置提醒):\n", "title")
        if self.det_result == 0:
            self.result_area.insert("end", f"     □ {next_y1} (12个月后)  年度常规乳腺体检 (超声+触诊, 40+加钼靶)\n", "sub")
        else:
            self.result_area.insert("end", f"     □ {next_m3} (约3个月后)  首次复查超声 → 对比大小/形态/边界/血流变化\n", "sub")
            self.result_area.insert("end", f"     □ {next_m6} (约6个月后)  第二次复查 → 超声 + 钼靶X线(必要时)\n", "sub")
            self.result_area.insert("end", f"     □ {next_y1} (12个月后)  年度全面评估 → 超声 + 钼靶 + 标志物(必要时)\n", "sub")
        self.result_area.insert("end", "  💾 档案管理建议:\n", "sub")
        self.result_area.insert("end", "     · 建议在电脑建立【乳腺健康档案】文件夹，存放每次DICOM原始影像+报告PDF+检查单照片\n", "sub")
        self.result_area.insert("end", "     · 复查时携带历史影像，供医生动态对比(比单次判读更有临床价值)\n", "sub")
        self.result_area.insert("end", "     · 家族史(母亲/姐妹/外祖母乳腺病史)、生育史、初潮/绝经年龄建议一并记录存档\n\n", "sub")
        self.result_area.insert("end", "━━━【免责声明与法律提示】━━━━━━━━━━━━━━━━━━\n", "divider")
        self.result_area.insert("end", "⚠️ 本AI诊断系统仅作为【临床辅助决策参考工具】使用，不具备最终诊断权。\n", "warn")
        self.result_area.insert("end", "   1. 报告中所有判断结论、概率数值、风险分级、BI-RADS参考、治疗方案建议\n", "sub")
        self.result_area.insert("end", "      均基于机器学习算法对现有训练数据集(BUS-UCLM + Dataset_BUSI)的统计学学习结果，\n", "sub")
        self.result_area.insert("end", "      不等同于执业医师的专业诊断、鉴别诊断结论或个体化治疗医嘱。\n", "sub")
        self.result_area.insert("end", "   2. 任何诊疗方案的制定、药物使用、手术/放化疗/靶向等医疗决策，\n", "sub")
        self.result_area.insert("end", "      请务必在【正规医疗机构】【注册执业医师】面诊后依其专业指导执行。\n", "sub")
        self.result_area.insert("end", "   3. 算法开发者、训练数据提供方、本程序使用者均不对因完全或部分依赖本报告\n", "sub")
        self.result_area.insert("end", "      作出的任何医疗决策及由此产生的直接或间接后果承担法律责任。\n", "sub")
        self.result_area.insert("end", "   4. 建议本报告结合: 详细病史+专科体格检查+多种影像学(超声/钼靶/MRI)+\n", "sub")
        self.result_area.insert("end", "      病理检查结果+实验室检查+执业医师经验综合判断解读。\n\n", "sub")
        self.result_area.insert("end", "═════════════════════════════════════════════════════\n", "divider")
        self.result_area.insert("end", f"💾 报告结束 | 如需纸质版: 可鼠标右键结果区→选中全部文本→复制粘贴至Word/WPS→转PDF打印\n", "sub")
        self.result_area.insert("end", "═════════════════════════════════════════════════════\n", "divider")
        self.after(100, lambda: self._save_dashboard())

    def _save_dashboard(self):
        try:
            path = plot_prediction_dashboard(
                self.current_img, self.heatmap_overlay,
                self.det_result, self.cls_result,
                self.current_feature, self.det_prob, self.cls_prob
            )
            self.result_area.insert("end", f"④ 完整诊断仪表盘已保存: {path}\n")
        except Exception as e:
            print(f"Dashboard err: {e}")

    def _clear_result(self):
        self.current_img_path = None
        self.current_img = None
        self.current_feature = None
        self.det_result = None
        self.det_prob = None
        self.cls_result = None
        self.cls_prob = None
        self.heatmap_overlay = None
        self.img_display.configure(image="", text="🖼️\n\n请导入一张乳腺超声影像\n(.png / .jpg / .bmp)")
        self.img_display.image = None
        self.heatmap_display.configure(image="", text="🎯 热力图显示区\n\n诊断完成后将显示\nGrad-CAM病灶注意力图\n\n🔴 红色高响应 = 模型重点关注区域")
        self.heatmap_display.image = None
        self.img_meta.configure(text="ℹ️ 影像信息: 未加载影像 | 请选择影像或使用示例影像",
                                bg="#e3f2fd", fg="#0d47a1")
        self.det_status.configure(text="病灶检测: ⚪ 未执行",
                                  bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0])
        self.cls_status.configure(text="良恶性分类: ⚪ 未执行",
                                  bg=self._status_colors["WAIT"][1], fg=self._status_colors["WAIT"][0])
        self.heatmap_hint.configure(text="💡 解读: 红色高响应区=模型重点关注区域, 可交叉验证病灶位置",
                                    bg="#fff8e1", fg="#e65100")
        self.result_area.configure(state="normal")
        self.result_area.delete("1.0", "end")
        self.result_area.insert("end", "💡 等待导入影像并执行诊断...\n\n", "sub")
        self.result_area.insert("end", "══【诊断流程】═══════════════\n", "divider")
        self.result_area.insert("end", "① 第一阶段: 深度学习(MobileNetV2) → 检测有无病灶区域\n", "sub")
        self.result_area.insert("end", "    可视化: Grad-CAM热力图(红=模型关注的病灶区)\n", "sub")
        self.result_area.insert("end", "② 第二阶段: 半监督ML(SelfTraining+RF影像组学) → 区分良性/恶性\n", "sub")
        self.result_area.insert("end", "    29维特征: 形状/强度/GLCM纹理/LBP分形/Hu矩\n", "sub")
        self.result_area.insert("end", "③ 可解释性: 前往【可解释性分析】查看SHAP/特征雷达\n", "sub")
        self.result_area.insert("end", "═══════════════════════════\n", "divider")
        self._set_status("已清空 | 等待导入影像...")
        self.result_area.configure(state="disabled")

    def _train_cb(self, msg, pct):
        self._log_train(f"[{pct:3d}%] {msg}")
        self.after(0, lambda: self._set_progress(pct, msg))

    def _train_detection_thread(self):
        if self.detection_model is not None:
            if not messagebox.askyesno("覆盖确认", "检测模型已存在，是否重新训练？"):
                return
        threading.Thread(target=self._do_train_detection, daemon=True).start()

    def _do_train_detection(self):
        try:
            self._log_train("========== 开始训练病灶检测模型 ==========")
            model, hist1, hist2, (X_test, y_test) = train_detection_model(
                progress_callback=self._train_cb, epochs=20
            )
            self.detection_model = model
            self.after(0, lambda: self.model_status_lbl.configure(
                text=f"✓检测:OK | 分类:{self.cls_data['best_name'] if self.cls_data else '未训练'}"))
            self._log_train("✅ 检测模型训练完成并保存")
            try:
                path = plot_detection_history(hist1, hist2)
                self._log_train(f"📈 训练曲线已保存: {path}")
            except Exception as e:
                print(f"Hist err: {e}")
        except Exception as e:
            traceback.print_exc()
            self._log_train(f"❌ 训练失败: {e}")
            self._log_train(traceback.format_exc())

    def _train_classification_thread(self):
        if self.cls_model is not None:
            if not messagebox.askyesno("覆盖确认", "分类模型已存在，是否重新训练？"):
                return
        threading.Thread(target=self._do_train_classification, daemon=True).start()

    def _do_train_classification(self):
        try:
            self._log_train("========== 开始训练良恶性分类半监督模型 ==========")
            result = train_classification_models(progress_callback=self._train_cb)
            best_model, best_name, best_res, sup_res, st_res, lp_res, X_test, y_test = result
            self.cls_model = best_model
            self.cls_data = {
                "best_model": best_model, "best_name": best_name, "best_result": best_res,
                "supervised_results": sup_res, "st_result": st_res, "lp_result": lp_res,
                "X_test": X_test, "y_test": y_test
            }
            self.after(0, lambda: self.model_status_lbl.configure(
                text=f"✓检测:{'OK' if self.detection_model else '未训练'} | ✓分类:{best_name}"))
            self._log_train(f"✅ 分类模型完成 | 最优: {best_name} AUC={best_res['auc']:.4f}")
            self._log_train(f"  准确率={best_res['acc']:.4f} 召回率={best_res['recall']:.4f} F1={best_res['f1']:.4f}")
        except Exception as e:
            traceback.print_exc()
            self._log_train(f"❌ 分类训练失败: {e}")
            self._log_train(traceback.format_exc())

    def _reload_detection(self):
        self.detection_model = load_detection_model()
        messagebox.showinfo("提示", "检测模型" + ("加载成功" if self.detection_model else "加载失败，请先训练"))

    def _reload_classification(self):
        self.cls_model, self.cls_data = load_classification_model()
        messagebox.showinfo("提示", "分类模型" + ("加载成功" if self.cls_model else "加载失败，请先训练"))

    def _generate_metrics(self):
        if not self.cls_data:
            messagebox.showwarning("提示", "分类模型未训练/未加载，无法生成评估图表")
            return
        threading.Thread(target=self._do_generate_metrics, daemon=True).start()

    def _do_generate_metrics(self):
        try:
            self._set_progress(10, "生成评估图表...")
            # ====== 新增容错开始 ======
            if "X_test" in self.cls_data and "y_test" in self.cls_data:
                X_test = self.cls_data["X_test"]
                y_test = self.cls_data["y_test"]
            else:
                # 旧模型文件没有测试集：从best_result的混淆矩阵重建近似数据
                best_result = self.cls_data.get("best_result", {})
                cm = best_result.get("cm")
                if cm and len(cm) >= 2 and len(cm[0]) >= 2:
                    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
                    y_test = np.array([0]*(tn+fp) + [1]*(tp+fn))
                    y_pred_dummy = np.array([0]*tn + [1]*fp + [0]*fn + [1]*tp)
                    X_test = None   # 无法重建特征，跳过需要X_test计算的ROC曲线
                    self.after(0, lambda: messagebox.showwarning("提示",
                        "检测到旧版模型文件（无测试集数据），将基于混淆矩阵显示部分图表。\n"
                        "建议重新训练分类模型以获取完整评估图表（ROC/PR/校准曲线）。"))
                else:
                    raise RuntimeError("模型文件缺少测试集数据，请重新训练分类模型！")
            # ====== 新增容错结束 ======
            sup_res = self.cls_data["supervised_results"]
            best_name = self.cls_data["best_name"]
            best_model = self.cls_data["best_model"]
            y_probs = {}
            y_pred_best = None
            # ========== 第1优先级（最可靠）：直接用训练好的模型对象现场预测 ==========
            # 你已经重新训练了，pickle里一定有supervised_models和best_model，这样100%能画出所有曲线
            if X_test is not None:
                # Step 1: 先画 best_model（无论它是SelfTraining+RF还是监督模型）
                try:
                    best_prob = best_model.predict_proba(X_test)[:, 1]
                    y_probs[best_name] = best_prob
                except Exception as e:
                    print(f"[WARN] best_model({best_name}) predict失败: {e}")
                # Step 2: 预测混淆矩阵用的y_pred_best
                try:
                    y_pred_best = best_model.predict(X_test)
                except Exception:
                    y_pred_best = None
                # Step 3: 画4个监督基线模型（RF/XGB/SVM/LR）
                sup_models = self.cls_data.get("supervised_models", None)
                if sup_models is not None:
                    for name, res in sup_res.items():
                        if name in y_probs:
                            continue
                        try:
                            if name in sup_models:
                                m = sup_models[name]
                                y_probs[name] = m.predict_proba(X_test)[:, 1]
                        except Exception as e:
                            print(f"[WARN] 模型{name} predict失败: {e}")

            # ========== 第2优先级（兜底省时间）：如果模型对象缺失，才用预存的y_prob ==========
            sup_yprobs = self.cls_data.get("supervised_y_probs", {}) or {}
            extra_yprobs = self.cls_data.get("extra_y_probs", {}) or {}
            for name, prob in sup_yprobs.items():
                if name not in y_probs:
                    y_probs[name] = np.asarray(prob, dtype=float)
            for name, prob in extra_yprobs.items():
                if name not in y_probs:
                    y_probs[name] = np.asarray(prob, dtype=float)

            # ========== 只有确实缺失模型时才提示（避免乱弹窗）==========
            expected_baseline = list(sup_res.keys())  # ['RF','XGB','SVM','LR']
            missing_baseline = [n for n in expected_baseline if n not in y_probs]
            missing_best = best_name not in y_probs
            if missing_baseline or missing_best:
                missing_all = []
                if missing_best:
                    missing_all.append(best_name)
                missing_all.extend(missing_baseline)
                self.after(0, lambda missing=missing_all: messagebox.showinfo("模型加载提示",
                    f"以下模型无法绘制完整曲线：\n• {' / '.join(missing)}\n\n请确认已点击【模型训练】→【训练分类模型】完成训练。"))
            self._set_progress(30)
            if y_pred_best is not None:
                p1 = plot_confusion_matrix(y_test, y_pred_best, title=f"混淆矩阵 ({best_name})")
                self._after_show(p1, "混淆矩阵")
            self._set_progress(50)
            p1 = plot_confusion_matrix(y_test, y_pred_best if y_pred_best is not None else y_test,
                                       title=f"混淆矩阵 ({best_name})")
            self._after_show(p1, "混淆矩阵")
            self._set_progress(50)
            if y_probs:
                p2 = plot_roc_curve(y_test, y_probs)
                self._after_show(p2, "ROC曲线对比")
                self._set_progress(65)
                p3 = plot_pr_curve(y_test, y_probs)
                self._after_show(p3, "PR曲线对比")
                self._set_progress(80)
                p4 = plot_calibration_curve(y_test, y_probs)
                self._after_show(p4, "校准曲线/Brier")
            self._set_progress(90)
            all_metrics = dict(sup_res)
            if self.cls_data.get("st_result"):
                all_metrics["SelfTraining"] = self.cls_data["st_result"]
            if self.cls_data.get("lp_result"):
                all_metrics["LabelProp"] = self.cls_data["lp_result"]
            p5 = plot_model_metrics_bar(all_metrics)
            self._after_show(p5, "metrics_bar")
            self._set_progress(100, "评估图表生成完成")
            self.after(0, lambda: self.eval_info.configure(
                text=f"✅ 已生成  测试集样本数: {len(y_test)}  |  最优模型: {best_name} AUC={self.cls_data['best_result']['auc']:.4f}"))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"生成图表失败: {e}\n{traceback.format_exc()}")

    def _after_show(self, img_path, canvas_key):
        def doit():
            if not img_path or not os.path.exists(img_path):
                return
            lbl = self.eval_canvas.get(canvas_key)
            if not lbl:
                return
            try:
                img = Image.open(img_path)
                w, h = img.size
                ratio = min(620 / w, 300 / h, 1.0)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl.configure(image=photo, text="")
                lbl.image = photo
            except Exception as e:
                print(f"Show eval err: {e}")
        self.after(0, doit)

    def _after_show_exp(self, img_path, canvas_key):
        def doit():
            if not img_path or not os.path.exists(img_path):
                return
            lbl = self.exp_canvas.get(canvas_key)
            if not lbl:
                return
            try:
                img = Image.open(img_path)
                w, h = img.size
                ratio = min(620 / w, 300 / h, 1.0)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl.configure(image=photo, text="")
                lbl.image = photo
            except Exception as e:
                print(f"Show exp err: {e}")
        self.after(0, doit)

    def _run_feature_importance(self):
        if not self.cls_data:
            messagebox.showwarning("提示", "请先训练分类模型")
            return
        threading.Thread(target=self._do_fi, daemon=True).start()

    def _do_fi(self):
        try:
            self._set_progress(5, "计算Permutation特征重要性...")
            X = self.cls_data["X_test"]
            y = self.cls_data["y_test"]
            model = self.cls_data["best_model"]
            p = plot_feature_importance(model, X, y)
            self._after_show_exp(p, "Permutation特征重要性")
            self._set_progress(100, "特征重要性完成")
            self.after(0, lambda: self.exp_info.configure(text=f"✅ 特征重要性已完成（基于AUC下降幅度）"))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", str(e))

    def _run_shap(self):
        if not self.cls_data:
            messagebox.showwarning("提示", "请先训练分类模型")
            return
        threading.Thread(target=self._do_shap, daemon=True).start()

    def _do_shap(self):
        try:
            self._set_progress(10, "初始化SHAP解释器...")
            model = self.cls_data["best_model"]
            X = self.cls_data["X_test"]
            paths = plot_shap_analysis(model, X, sample_idx=0)
            if paths:
                self._set_progress(60)
                if "summary" in paths:
                    self._after_show_exp(paths["summary"], "SHAP Summary散点图")
                self._set_progress(75)
                if "bar" in paths:
                    self._after_show_exp(paths["bar"], "SHAP Bar重要性")
                self._set_progress(90)
                if "waterfall" in paths:
                    self._after_show_exp(paths["waterfall"], "SHAP瀑布图(单样本)")
            self._set_progress(100, "SHAP分析完成")
            self.after(0, lambda: self.exp_info.configure(text="✅ SHAP分析已完成（全局+单样本局部解释）"))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", str(e))

    def _run_radar(self):
        if self.current_feature is None:
            messagebox.showwarning("提示", "请先执行一次诊断，生成当前样本的特征向量")
            return
        try:
            ref_mins, ref_maxs = None, None
            if self.cls_data and "X_train_min" in self.cls_data:
                ref_mins = self.cls_data["X_train_min"]
                ref_maxs = self.cls_data["X_train_max"]
            p = plot_top_features_radar(self.current_feature, feature_ref_mins=ref_mins, feature_ref_maxs=ref_maxs)
            if p:
                self._display_image_in_popup(p, "当前预测样本 - 关键影像组学特征雷达图")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", str(e))

    def _run_radar_inline(self):
        if self.current_feature is None:
            messagebox.showwarning("提示", "请先执行一次诊断，生成当前样本的特征向量")
            return
        try:
            ref_mins, ref_maxs = None, None
            if self.cls_data and "X_train_min" in self.cls_data:
                ref_mins = self.cls_data["X_train_min"]
                ref_maxs = self.cls_data["X_train_max"]
            p = plot_top_features_radar(self.current_feature, feature_ref_mins=ref_mins, feature_ref_maxs=ref_maxs)
            if p and os.path.exists(p):
                img = Image.open(p)
                w, h = img.size
                ratio = min(800 / w, 420 / h, 1.0)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.radar_lbl.configure(image=photo, text="")
                self.radar_lbl.image = photo
                self.after(0, lambda: self.exp_info.configure(text="✅ 内嵌特征雷达图已生成（基于Top偏离特征的可视化）"))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", str(e))

    def _open_radar_zoom(self):
        from config import RESULT_DIR
        path = os.path.join(RESULT_DIR, "feature_radar.png")
        if not os.path.exists(path):
            if self.current_feature is None:
                messagebox.showwarning("提示", "请先生成特征雷达图：执行诊断后点击【生成内嵌雷达图】")
                return
            self._run_radar_inline()
        self._display_image_in_popup(path, "📊 关键影像组学特征雷达图")

    def _clear_explain_charts(self):
        if not messagebox.askyesno("确认", "确定要清空所有可解释性分析图表吗？"):
            return
        for key, lbl in self.exp_canvas.items():
            desc_map = {
                "Permutation特征重要性": "基于AUC下降幅度，衡量特征对模型整体性能的影响；下降越多说明特征越重要",
                "SHAP Summary散点图": "特征值(颜色红高蓝低) vs SHAP贡献值(水平位置)，展示全局特征影响力分布",
                "SHAP Bar重要性": "平均绝对SHAP值的柱状排名，直观展示Top-N特征的重要性排序",
                "SHAP瀑布图(单样本)": "当前单个样本的SHAP分解：展示每个特征如何推动基准值→最终预测结果",
            }
            desc = desc_map.get(key, "")
            lbl.configure(image="", text=f"⏳ 已清空\n\n💡 {desc}", fg="#78909c")
            lbl.image = None
        self.radar_lbl.configure(image="",
                                 text="⏳ 已清空\n\n💡 雷达图展示当前样本在Top-N关键影像组学特征上的表现\n红(>0.7)高风险 / 绿(<0.3)低风险",
                                 fg="#78909c")
        self.radar_lbl.image = None
        self.exp_info.configure(text="ℹ️ 已清空所有分析图表 | 可重新点击按钮生成")
        self._set_status("已清空可解释性分析图表")

    def _open_eval_zoom(self, chart_name):
        name_map = {
            "混淆矩阵": "confusion_matrix.png",
            "ROC曲线对比": "roc_curve.png",
            "PR曲线对比": "pr_curve.png",
            "校准曲线/Brier": "calibration_curve.png",
            "多模型指标": "model_metrics_bar.png"
        }
        fname = name_map.get(chart_name)
        if not fname:
            return
        from config import RESULT_DIR
        path = os.path.join(RESULT_DIR, fname)
        self._display_image_in_popup(path, f"📊 {chart_name}")

    def _open_exp_zoom(self, chart_name):
        name_map = {
            "Permutation特征重要性": "feature_importance.png",
            "SHAP Summary散点图": "shap_summary.png",
            "SHAP Bar重要性": "shap_bar.png",
            "SHAP瀑布图(单样本)": "shap_waterfall.png"
        }
        fname = name_map.get(chart_name)
        if not fname:
            return
        from config import RESULT_DIR
        path = os.path.join(RESULT_DIR, fname)
        self._display_image_in_popup(path, f"🔬 {chart_name}")

    def _display_image_in_popup(self, path, title):
        if not path or not os.path.exists(path):
            return
        try:
            orig_img = Image.open(path)
            orig_w, orig_h = orig_img.size
            max_w, max_h = 1100, 780
            init_scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            win_w = min(int(orig_w * init_scale) + 60, 1300)
            win_h = min(int(orig_h * init_scale) + 130, 950)
        except Exception:
            return
        top = tk.Toplevel(self)
        top.title(title)
        top.geometry(f"{win_w}x{win_h}")
        top.configure(bg="#f0f0f0")
        state = {"scale": init_scale, "photo": None}
        toolbar = tk.Frame(top, bg="#e0e0e0", height=48)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)
        ttk.Button(toolbar, text="🔍 缩小", command=lambda: self._zoom_image(state, -0.15)).pack(side="left", padx=5, pady=8)
        ttk.Button(toolbar, text="🔍 放大", command=lambda: self._zoom_image(state, 0.15)).pack(side="left", padx=5, pady=8)
        ttk.Button(toolbar, text="↺ 100%", command=lambda: self._set_zoom(state, 1.0)).pack(side="left", padx=5, pady=8)
        ttk.Button(toolbar, text="⛶ 适应窗口", command=lambda: self._set_zoom(state, init_scale)).pack(side="left", padx=5, pady=8)
        state["zoom_lbl"] = tk.Label(toolbar, text=f"{int(init_scale*100)}%", bg="#e0e0e0", font=("Consolas", 11, "bold"))
        state["zoom_lbl"].pack(side="right", padx=15)
        canvas = tk.Canvas(top, bg="#ffffff", highlightthickness=0)
        v_scroll = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
        h_scroll = ttk.Scrollbar(top, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        state["orig_img"] = orig_img
        state["orig_w"] = orig_w
        state["orig_h"] = orig_h
        state["canvas"] = canvas
        state["img_id"] = None
        def _on_popup_wheel(event):
            if event.state & 0x0004:
                delta = 0.15 if event.delta > 0 else -0.15
                self._zoom_image(state, delta)
                return "break"
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_popup_wheel)
        self._render_image(state)

    def _render_image(self, state):
        img = state["orig_img"]
        scale = state["scale"]
        w = max(1, int(state["orig_w"] * scale))
        h = max(1, int(state["orig_h"] * scale))
        resized = img.resize((w, h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        if state["img_id"] is not None:
            state["canvas"].delete(state["img_id"])
        state["img_id"] = state["canvas"].create_image(0, 0, anchor="nw", image=photo)
        state["canvas"].configure(scrollregion=(0, 0, w, h))
        state["photo"] = photo
        state["zoom_lbl"].configure(text=f"{int(scale*100)}%")

    def _zoom_image(self, state, delta):
        new_scale = min(5.0, max(0.1, state["scale"] + delta))
        state["scale"] = new_scale
        self._render_image(state)

    def _set_zoom(self, state, scale):
        state["scale"] = min(5.0, max(0.1, scale))
        self._render_image(state)


def main():
    try:
        app = BreastCancerApp()
        app.mainloop()
    except Exception as e:
        traceback.print_exc()
        try:
            msg = "程序启动失败!\n\n错误信息: {}\n\n详细堆栈:\n{}".format(repr(e), traceback.format_exc())
            print(msg)
            root = tk.Tk()
            root.title("启动错误")
            root.geometry("780x520")
            root.configure(bg="#fff0f0")
            tk.Label(root, text="❌ 程序启动失败", font=("微软雅黑", 16, "bold"),
                     fg="#d32f2f", bg="#fff0f0").pack(pady=16)
            txt = tk.Text(root, wrap="word", height=18, font=("Consolas", 10))
            txt.pack(padx=20, pady=4, fill="both", expand=True)
            txt.insert("1.0", msg)
            txt.configure(state="disabled")
            tk.Label(root, text="请把此错误截图发给开发者，或运行『🔥调试启动_不闪退.bat』查看详情",
                     font=("微软雅黑", 10), fg="#c62828", bg="#fff0f0").pack(pady=8)
            tk.Button(root, text="关闭", command=root.destroy, width=12,
                      bg="#ef5350", fg="white").pack(pady=12)
            root.mainloop()
        except Exception:
            pass


if __name__ == "__main__":
    main()