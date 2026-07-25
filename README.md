# 乳腺超声影像AI智能诊断系统

> Breast Ultrasound Image AI Intelligent Diagnosis System
> 基于两阶段深度学习+半监督学习+影像组学的端到端乳腺病灶智能检测与良恶性分类系统

---

## 项目简介

针对基层医院乳腺癌超声筛查缺乏专业影像科医生的痛点，本系统实现两阶段智能诊断流水线：

1. **第一阶段 - 病灶检测**：MobileNetV2 迁移学习 → 判断「正常 / 有病灶」
2. **第二阶段 - 良恶性分类**：影像组学特征 + SelfTraining 半监督学习 → 判断「良性 / 恶性」
3. **临床决策支持**：BI-RADS 风险分级 + 完整诊断报告 + 个性化治疗方案 + 随访计划
4. **可解释性分析**：Grad-CAM 热力图 + SHAP 归因分析 + 特征雷达图，让 AI 决策透明

### 系统核心特性

| 特性 | 说明 |
|-----|------|
| **抗数据泄漏** | 按患者 ID 分组划分训练 / 测试集，彻底杜绝患者级泄漏 |
| **半监督学习** | 少量有标签 + 大量无标签伪标签，AUC 相比纯监督提升约 6% |
| **概率校准** | SelfTraining+RF 的 Brier 损失最低，临床预测最诚实可靠 |
| **极低漏诊率** | 漏诊率 < 误诊率，符合癌症筛查「宁可错杀、不能漏诊」的安全原则 |
| **完整 GUI** | Tkinter 桌面应用，中文界面，双击启动无需命令行 |
| **中文路径支持** | PIL 读取 + cv2 兼容，彻底解决 OpenCV 中文路径崩溃问题 |

---

## 项目结构与各文件作用
```text
practice/                              # ========== 【GitHub 仓库根目录】 ==========
├── breast_cancer_app/                 # ✅ 【软件核心】桌面程序代码（和 practice_data 同级分开上传）
│   ├── gui_app.py                     # 🖥️ 主界面 (Tkinter GUI，约4000行)
│   │                                    - 三大Tab：影像诊断 / 模型训练 / 诊断与评估
│   │                                    - 重点函数：_show_predict_result (生成完整诊断报告)
│   │                                    - 重点函数：_do_generate_metrics (生成ROC/PR等评估图表)
│   │
│   ├── config.py                      # ⚙️ 全局配置文件（所有可调参数、路径、常量的入口）
│   │                                    - 路径配置：DATA_DIR (指向practice_data)、MODEL_DIR…
│   │                                    - 超参数：RANDOM_SEED、IMG_SIZE、BATCH_SIZE、EPOCHS…
│   │                                    - 特征名列表：FEATURE_NAMES（29维影像组学特征名）
│   │
│   ├── data_loader.py                 # 📦 数据加载 + 划分（反泄漏设计的核心）
│   │                                    - load_all_data()：双数据集(BUSI+BUS-UCLM)统一加载
│   │                                    - _group_split_by_patients()：★按患者ID分组划分★ 防泄漏
│   │                                    - split_detection_data()：检测任务 train/test 划分
│   │                                    - split_classify_data()：分类任务 train/test/unlabeled 划分
│   │
│   ├── detection_model.py             # 🔎 第一阶段：MobileNetV2 病灶检测模型
│   │                                    - build_detection_cnn()：MobileNetV2 + 迁移学习 + 自定义分类头
│   │                                    - data_generator()：在线数据增强 + 样本权重（解决类别不平衡）
│   │                                    - train_detection_model()：冻结骨干 → 解冻微调 两阶段训练
│   │
│   ├── classification_model.py        # 🧬 第二阶段：良恶性分类（6模型对比 + 半监督）
│   │                                    - build_supervised_model()：4基线 RF/XGB/SVM/LR
│   │                                    - build_semisupervised_model()：SelfTraining + LabelPropagation
│   │                                    - train_classification_models()：★5折CV选最优 + 测试集仅报告★
│   │                                    - save_classification_model()：pickle 序列化模型+测试集+预测概率
│   │
│   ├── feature_extractor.py           # 📊 29维影像组学特征提取
│   │                                    - 形状16维：area_ratio/perimeter/circularity/Hu不变矩×7…
│   │                                    - 强度6维：mean_intensity/skewness/kurtosis/entropy…
│   │                                    - 纹理7维：GLCM 5项 + lbp_entropy + texture_uniformity
│   │
│   ├── segmentation.py                # ✂️ 病灶自动分割模块
│   │                                    - 自适应阈值法 + 形态学开闭运算 → 生成病灶 RoI Mask
│   │                                    - 分割失败兜底策略：全图做 Mask + 提示信息
│   │
│   ├── visualization.py               # 📈 所有可视化图表生成
│   │                                    - plot_roc_curve / plot_pr_curve / plot_calibration_curve
│   │                                    - plot_confusion_matrix / plot_model_metrics_bar
│   │                                    - plot_top_features_radar()：★特征雷达图（归一化修复版）★
│   │                                    - Permutation 特征重要性盒图
│   │
│   ├── shap_explainer.py              # 🔬 SHAP 可解释性分析模块
│   │                                    - SHAP summary_plot / bar_plot / waterfall_plot
│   │
│   ├── saved_models/                  # 💾 训练好的模型文件（首次训练后自动生成）
│   │   ├── detection_model.h5         #    MobileNetV2 检测模型权重
│   │   └── classification_model.pkl   #    最优分类模型 + 测试集 + 预测概率 + 评估结果
│   │
│   ├── results/                       # 🖼️ 评估图表输出目录（点击按钮后自动生成）
│   │
│   ├── requirements.txt               # 📋 Python 依赖清单（pip install -r requirements.txt）
│   └── ✅双击这个启动图形界面.bat      # 🚀 Windows 用户一键启动脚本
│
├── practice_daima/                    # 📓 【教学练习】Jupyter Notebook（与软件代码对比用）
│   └── 阶段二练习.ipynb               #    纯监督学习 + GridSearchCV 嵌套5折调参完整流程
│
└── practice_data/                     # 📁 【数据集】（与 breast_cancer_app 同级，分开上传）
    ├── Dataset_BUSI/                  #    BUSI 公开乳腺超声数据集
    │   └── Dataset_BUSI_with_GT/
    │       ├── benign/                #       良性图像 + 对应 Mask
    │       ├── malignant/             #       恶性图像 + 对应 Mask
    │       └── normal/                #       正常图像 + 对应 Mask
    └── BUS-UCLM/                      #    BUS-UCLM 公开乳腺超声数据集
        ├── images/                    #       超声图像文件
        └── INFO.csv                   #       患者ID + 标签 + 元信息 CSV
```
---

## 整体思路与诊断流水线

      ┌──────────────────────────────────────────────────────────┐
      │               输入：患者乳腺超声图像                       │
      └──────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
      ┌──────────────────────────────────────────────────────────┐
      │  第一阶段：MobileNetV2 病灶检测（有病灶 / 正常二分类）     │
      │  输入：224×224 RGB 图像 ｜ 输出：有病灶概率 P_detection   │
      └──────────────────────────────┬───────────────────────────┘
                                     │
                   ┌─────────────────┴───────────────────┐
                   │                                     │
                   ▼                                     ▼
      ┌──────────────────────┐            ┌──────────────────────────────┐
      │ P<阈值 → 正常(无病灶) │            │ P≥阈值 → 判定为「有病灶」       │
      │ BI-RADS 1 级          │            │ 进入第二阶段                  │
      │ 建议：年度常规随访     │            └──────────────┬───────────────┘
      └──────────────────────┘                           │
                                                         ▼
                                      ┌────────────────────────────────────┐
                                      │ 自动分割模块：阈值法 + 形态学处理    │
                                      │ 输出：病灶区域 Mask (RoI)           │
                                      └──────────────────┬─────────────────┘
                                                         │
                                                         ▼
                                      ┌────────────────────────────────────┐
                                      │ 影像组学特征提取（基于 Mask + 原图） │
                                      │ 形状16 + 强度6 + 纹理7 = 共29维     │
                                      └──────────────────┬─────────────────┘
                                                         │
                                                         ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │      第二阶段：SelfTraining+RF 半监督模型 良恶性分类               │
      │      输入：29维特征向量 ｜ 输出：恶性概率 P_malignant                │
      └──────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │  综合输出：完整临床诊断报告                                          │
      │  ① P_malignant + BI-RADS 分级（2/3/4a/4b/4c/5/6）                   │
      │  ② 推荐治疗方案：穿刺活检 / 微创手术 / 随访 / 转上级医院              │
      │  ③ 随访日历：下次复查时间 + 检查项目                                  │
      │  ④ 风险评估 + 家族史提醒                                              │
      │  ⑤ 可解释性：特征雷达图 + SHAP归因 + Grad-CAM热力图                  │
      └────────────────────────────────────────────────────────────────────┘

    
---

## 核心原理详解

### 1️第一阶段：MobileNetV2 病灶检测

#### 为什么选择 MobileNetV2？

| 核心创新 | 原理简述 | 对本项目的价值 |
|---------|---------|-------------|
| **深度可分离卷积** | 把标准 3×3 卷积拆成「3×3 通道内卷积 + 1×1 跨通道卷积」，计算量仅为普通卷积的 1/5~1/9 | 基层医院老旧 CPU 电脑也能实现单张 < 0.5 秒推理 |
| **倒残差结构** | 低维 → 1×1 升维 → 3×3 DWConv → 1×1 降维 → 残差连接 | 在低维空间做特征，保留超声图像的细小病灶细节 |
| **线性瓶颈** | 最后降维层使用线性激活（不接 ReLU），避免低维信息丢失 | 小型边界模糊病灶的信息不会被激活函数抹掉 |

#### 两阶段训练策略

**阶段 1（Epoch 0 ~ 16）：**
*冻结骨干 + 训练分类头 · MobileNetV2 ImageNet 预训练权重全部冻结 · 只学习顶层 2 层 Dense + Dropout · 学习率：1e-4*

**阶段 2（Epoch 17 ~ 26）：**
*解冻骨干后 60 层 + 微调 · 学习率降为原来 1/10（1e-5），防止破坏预训练特征 · 用 EarlyStopping + ModelCheckpoint 监控 val_auc，取最优 epoch*

---

### 第二阶段：半监督学习 Self-Training + 影像组学

#### 29 维影像组学特征分类

| 大类 | 维度 | 代表特征 | 临床意义 |
|-----|-----|---------|---------|
| 形状特征 | 16 维 | circularity, solidity, eccentricity, Hu不变矩×7 | 恶性结节：不规则形、分叶毛刺、纵横比 > 1 |
| 强度特征 | 6 维 | skewness, kurtosis, entropy, mean/std | 恶性结节：后方回声衰减、灰度分布极不均匀 |
| 纹理特征 | 7 维 | GLCM_contrast, GLCM_homogeneity, lbp_entropy | 恶性结节：毛刺导致局部纹理混乱，同质性低 |

#### Self-Training 半监督学习流程（归纳式，可部署）

第 0 步：准备数据 有标签 L：160 例（医生标注过良性/恶性） 无标签 U：240 例（只拿图，不知道标签 → 模拟真实场景无标注数据）

第 1 步：初始模型 只用 L 训练一个 RandomForest 基线模型

第 2 ~ 15 步：伪标签迭代（threshold = 0.85） ┌─────────────────────────────────────────────────────┐ │ ① 用当前模型预测 U 中所有样本的恶性概率 P │ │ ② P ≥ 0.85 → 打上「伪恶性」标签 │ │ ③ P ≤ 0.15 → 打上「伪良性」标签 │ │ ④ 把高置信度的伪标签样本合并进训练集 L' │ │ ⑤ 用 L + 伪标签样本 重新训练模型，更新权重 │ │ ⑥ 回到 ①，直到满足 max_iter=15 或没有高置信样本 │ └─────────────────────────────────────────────────────┘

最终效果：模型见过 400 例数据的分布 → 比纯监督 160 例泛化更好


---

## ⚙️ 使用时需要调整的文件 & 可调参数大全

---

### 文件 1：`config.py` —— 【最常改，所有配置入口】

| 参数名称 | 默认值 | 调整建议与意义 |
|---------|-------|-------------|
| `RANDOM_SEED` | 42 | 固定即可，改动会导致结果不可复现 |
| `DATA_DIR` | `"../practice_data"` | 数据集没放默认路径时，**必须改**这个绝对路径 |
| `BUSI_DIR` / `BUS_UCLM_DIR` | 在 `DATA_DIR` 下 | 子文件夹命名不一致时改这两个 |
| `MODEL_DIR` | `"./saved_models"` | 模型保存目录，一般不用改 |
| `RESULT_DIR` | `"./results"` | 图表输出目录，一般不用改 |
| `IMG_SIZE` | `224` | MobileNetV2 输入尺寸；改 256/384 → 精度↑ 推理速度↓↓ |
| `BATCH_SIZE` | `16` | 显存 ≥ 8G 可调到 32；小显存或 CPU 改 8 |
| `EPOCHS_DETECTION` | `26` | 总训练轮数（16 冻 + 10 微）；过拟合减到 20，欠拟合加到 32 |

---

### 文件 2：`classification_model.py` —— 【模型超参数】

#### `build_supervised_model()` 内 4 基线模型：

| 模型 | 参数名 | 默认值 | 调参意义 |
|-----|-------|-------|---------|
| **RF** | n_estimators | 400 | 树的数量：↑1000 → 更稳更慢；↓100 → 更快 |
| RF | max_depth | 14 | 单棵树最大深度：↓10 → 防过拟合；↑16 → 易过拟合 |
| RF | min_samples_split | 4 | 分裂最小样本数：↑防过拟合 |
| **XGB** | n_estimators | 300 | GBDT 树数：小数据集 ↓100 防过拟合 |
| XGB | max_depth | 6 | 树深：小数据集 ↓3 防过拟合 |
| XGB | learning_rate | 0.08 | 学习率：配合 n_estimators 反向调整 |
| XGB | subsample | 0.85 | 子采样比例：0.7~0.9 防过拟合 |
| **SVM** | C | 10.0 | 正则化惩罚系数：↑易过拟合，↓易欠拟合 |
| **LR** | C | 1.0 | 正则化系数：↑减少正则化，↓增强正则化 |

#### `build_semisupervised_model()` 内半监督参数：

| 参数名 | 默认值 | 调参意义 |
|-------|-------|---------|
| `threshold` | 0.85 | 伪标签置信度阈值：↑0.9 → 伪标签少但准；↓0.8 → 多但可能引入噪声 |
| `max_iter` | 15 | SelfTraining 最大迭代轮次，一般 5~20 内收敛 |
| 基模型 n_estimators | 300 | 半监督基 RF 比监督版略小（防伪标签噪声放大） |
| 基模型 max_depth | 12 | 比监督版略浅（防伪标签噪声过拟合） |

---

### 文件 3：`data_loader.py` —— 【数据划分比例】

#### `split_detection_data(records, test_size=0.2)`
- `test_size=0.2`：检测任务测试集比例（20%），一般不用改

#### `split_classify_data(records, test_size=0.2, unlabeled_ratio=0.6)`
- `test_size=0.2`：分类任务测试集比例（有标签的 20% 做测试）
- `unlabeled_ratio=0.6`：**模拟无标签集的比例**
  - 含义：从全部有标签样本里拿 60%「隐藏标签」当作伪无标签
  - 想对比纯监督 vs 半监督的差异可以调整：0.4 / 0.6 / 0.8

---

### 文件 4：`detection_model.py` —— 【检测训练细节】

#### 两阶段微调控制（`train_detection_model` 内）：
- `unfreeze_layers`：解冻最后多少层（默认后 60 层）
- 阶段 1 学习率 `1e-4`，阶段 2 学习率 `1e-5`（一般不用改）

---

## 环境与依赖需求

### Python 版本
- **推荐版本**：Python 3.9 / 3.10 / 3.11（3.12+ TensorFlow 可能兼容性差）
- **环境管理工具**：强烈推荐 Anaconda / Miniconda 创建虚拟环境

### 核心依赖版本要求（`requirements.txt` 已写好）

| 包名 | 最低版本 | 作用 |
|-----|---------|------|
| tensorflow | 2.10.0 | MobileNetV2 深度学习训练推理 |
| scikit-learn | 1.2.0 | 传统 ML 模型 + SelfTraining + 评估指标 |
| scikit-image | 0.20.0 | GLCM 纹理特征、Hu 不变矩、局部二值模式 |
| opencv-python | 4.5.0 | 图像读写、形态学处理（注意中文路径用 PIL 兜底） |
| pandas | 1.5.0 | CSV 元信息加载、DataFrame 操作 |
| numpy | 1.23.0 | 所有数值计算基础 |
| matplotlib | 3.7.0 | 所有图表绘制（ROC/PR/校准/雷达…） |
| seaborn | 0.12.0 | 热力图、特征重要性美化 |
| shap | 0.41.0 | 可选，SHAP 可解释性分析 |
| pillow | 9.0.0 | PIL 读取中文路径图像 |

---

## 完整使用流程

### ① 数据集准备（首次使用必做）

- 按照「项目结构」的目录结构摆放好
- 修改 `config.py` 的 `DATA_DIR` 如果路径不同

### ② 启动软件
- Windows：双击 `practice\breast_cancer_app\双击这个启动图形界面.bat`
- 其他：`python gui_app.py`

### ③ 首次训练模型（每次换机器必做）
GUI → 【模型训练】Tab → 点【训练检测模型】（MobileNetV2 迁移学习，CPU 约 15~30 分钟） → 点【训练分类模型】（6 模型对比 + 5 折 CV，约 3~5 分钟） → 进度条 100% 后，saved_models/ 下会生成两个模型文件：`detection_model.h5` 和 `classification_model.h5`

### ④ 日常影像诊断
GUI → 【影像诊断】Tab → 点【选择影像】选自己的超声图，或点【使用示例影像】 → 点【开始智能诊断】 → 右侧查看完整报告： · 检测结论 + 良恶性概率 · BI-RADS 分级 + 风险评估 · 治疗方案建议 + 下次随访时间 · 特征雷达图 + Grad-CAM 热力图（可解释性）

### ⑤ 模型评估图表
GUI → 【诊断与评估】Tab → 点【生成评估图表】 → 自动弹出 6 张图： · 混淆矩阵（最优模型） · ROC 曲线对比（5~6 模型） · PR 曲线对比 · 校准曲线 + Brier 分数 · 各模型 AUC / Recall / F1 柱状图 · 前 15 特征重要性（Permutation）

---

## 预期结果范围（参考）

| 指标 | SelfTraining+RF（最终部署） | 纯监督 RF（基线） |
|-----|---------------------------|-----------------|
| Test AUC | ~0.95 | ~0.94 |
| Brier 损失 | ~0.06 | ~0.063 |
| 召回率（漏诊反向） | ~96.8% | ~95.9% |
| 特异度 | ~95.6% | ~94.5% |
| 混淆矩阵 | TN≈43 FP≈2 FN≈1 TP≈30 | TN≈42 FP≈3 FN≈2 TP≈29 |

> 注：结果随数据划分随机波动 ±1%，固定 RANDOM_SEED=42 可 100% 复现。

---

## 复现性保障机制

| 层级 | 保障方式 | 位置 |
|-----|---------|-----|
| Python 哈希 | `PYTHONHASHSEED=42`（所有 import 前设置） | detection_model.py 最开头 |
| 全局随机 | `random.seed(42)`, `np.random.seed(42)` | detection / classification 开头 |
| 数据划分 | `np.random.RandomState(42)` 独立实例 | data_loader.py `_group_split_by_patients` |
| sklearn 模型 | 所有分类器 `random_state=42` | classification_model.py `build_*` |
| TensorFlow | `tf.random.set_seed(42)` + deterministic ops | detection_model.py TF import 后 |
| 交叉验证 | `StratifiedKFold(..., random_state=42)` | classification_model.py 训练流程 |

---

## 常见问题排查

| 问题现象 | 原因 | 解决办法 |
|---------|-----|---------|
| 诊断结果面板空白无文字 | `result_area` 设了 disabled 无法 insert | 已修复，确保 `_show_predict_result` 操作前 state=normal |
| 雷达图只有 1 个特征 = 1.0，其他全 0 | 归一化算法 top_k 单独子归一化 | 已重写 visualization.py，逐特征独立映射 |
| ROC 曲线只有半监督一条线 | pickle 里没存基线模型 / 预测概率 | 重新点【训练分类模型】即可 |
| 检测模型 class_weight 报错 | 生成器输入不支持 class_weight 参数 | 已修复：generator 返回 (X, y, sample_weight) |
| AUC = 1.0（XGB/RF） | 测试集上重新训练模型作弊（旧 BUG） | 已修复：画图直接用训练好的对象 predict，不再 fit X_test |
| 中文路径 cv2.imread 返回 None | OpenCV 不支持中文路径 | 已修复：用 PIL.Image.open 读取，再转 ndarray |

---

## 免责声明

本项目仅用于学术研究、教学和个人学习用途，**不能用于任何临床诊断或医疗决策**。
乳腺疾病的最终诊断请务必前往正规三甲医院，由专业影像科医生 + 病理活检确诊。
本文不涉及原理介绍，仅作为实际应用示例。

---

