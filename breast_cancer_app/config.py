import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_1 = os.path.dirname(BASE_DIR)
PARENT_2 = os.path.dirname(PARENT_1)

def _find_practice_data():
    candidates = [
        os.path.join(PARENT_1, "practice_data"),
        os.path.join(PARENT_2, "practice_data"),
        os.path.join(BASE_DIR, "practice_data"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "BUS-UCLM", "INFO.csv")):
            return c
    print(f"[WARN] 未自动定位 practice_data，已假设路径: {candidates[0]}")
    return candidates[0]

DATA_DIR = _find_practice_data()
BUS_UCLM_DIR = os.path.join(DATA_DIR, "BUS-UCLM")
BUS_UCLM_IMG = os.path.join(BUS_UCLM_DIR, "images")
BUS_UCLM_CSV = os.path.join(BUS_UCLM_DIR, "INFO.csv")

def _find_busi():
    p1 = os.path.join(DATA_DIR, "Dataset_BUSI", "Dataset_BUSI_with_GT")
    p2 = os.path.join(DATA_DIR, "Dataset_BUSI")
    if os.path.isdir(p1):
        return p1
    return p2

BUSI_DIR = _find_busi()
BUSI_BENIGN = os.path.join(BUSI_DIR, "benign")
BUSI_MALIGNANT = os.path.join(BUSI_DIR, "malignant")
BUSI_NORMAL = os.path.join(BUSI_DIR, "normal")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_DETECTION = 20
RANDOM_SEED = 42

LABEL_MAP_DETECTION = {0: "正常(无病灶)", 1: "有病灶(异常)"}
LABEL_MAP_CLASSIFY = {0: "良性(Benign)", 1: "恶性(Malignant)"}

FEATURE_NAMES = [
    "area_ratio", "perimeter", "circularity", "mean_intensity", "std_intensity",
    "contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation",
    "skewness", "kurtosis", "entropy", "compactness", "solidity",
    "eccentricity", "equivalent_diameter", "major_axis", "minor_axis",
    "lbp_entropy", "texture_uniformity", "hu_moment_1", "hu_moment_2",
    "hu_moment_3", "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"
]