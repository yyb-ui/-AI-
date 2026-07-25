import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from config import (
    BUS_UCLM_IMG, BUS_UCLM_CSV, BUSI_BENIGN, BUSI_MALIGNANT, BUSI_NORMAL,
    IMG_SIZE, RANDOM_SEED
)


def _extract_patient_id_busi(img_path):
    """从BUSI文件名提取患者ID: benign (1).png -> busi_benign_1"""
    fname = os.path.basename(img_path)
    folder = os.path.basename(os.path.dirname(img_path))
    # 提取括号中的数字
    import re
    m = re.search(r'\((\d+)\)', fname)
    if m:
        return f"busi_{folder}_{m.group(1)}"
    return f"busi_{folder}_{fname}"


def _extract_patient_id_uclm(img_path):
    """从BUS-UCLM文件名提取患者ID: ALWI_000.png -> uclm_ALWI"""
    fname = os.path.basename(img_path)
    # 取下划线前的前缀作为患者标识
    prefix = fname.split('_')[0] if '_' in fname else fname.split('.')[0]
    return f"uclm_{prefix}"


def _pil_read_cv2(img_path, gray=False):
    """Pillow读取中文路径 → 转 OpenCV 格式(BGR/RGB uint8)，彻底解决cv2.imread中文路径BUG"""
    try:
        if not os.path.exists(img_path):
            return None
        pil_img = Image.open(img_path)
        if gray:
            pil_img = pil_img.convert("L")
            arr = np.array(pil_img, dtype=np.uint8)
            return arr
        else:
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img, dtype=np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[PIL_READ_FAIL] {img_path}: {e}")
        return None


def load_bus_uclm_data():
    import csv
    records = []
    with open(BUS_UCLM_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            img_name = row["Image"]
            label_str = row["Label"]
            img_path = os.path.join(BUS_UCLM_IMG, img_name)
            if not os.path.exists(img_path):
                continue
            if label_str == "Normal":
                label_det = 0
                label_cls = None
            elif label_str == "Benign":
                label_det = 1
                label_cls = 0
            elif label_str == "Malignant":
                label_det = 1
                label_cls = 1
            else:
                continue
            patient_id = _extract_patient_id_uclm(img_path)
            records.append({
                "img_path": img_path,
                "mask_path": None,
                "label_detection": label_det,
                "label_classify": label_cls,
                "source": "BUS-UCLM",
                "patient_id": patient_id
            })
    return records


def load_busi_data():
    records = []
    for folder, lbl_det, lbl_cls in [
        (BUSI_NORMAL, 0, None),
        (BUSI_BENIGN, 1, 0),
        (BUSI_MALIGNANT, 1, 1)
    ]:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if "mask" in f.lower():
                continue
            if not (f.lower().endswith(".png") or f.lower().endswith(".jpg")):
                continue
            img_path = os.path.join(folder, f)
            base = os.path.splitext(f)[0]
            mask_candidates = [
                os.path.join(folder, base + "_mask.png"),
                os.path.join(folder, base + "_mask_1.png")
            ]
            mask_path = None
            for mc in mask_candidates:
                if os.path.exists(mc):
                    mask_path = mc
                    break
            patient_id = _extract_patient_id_busi(img_path)
            records.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "label_detection": lbl_det,
                "label_classify": lbl_cls,
                "source": "BUSI",
                "patient_id": patient_id
            })
    return records


def load_all_data():
    data1 = load_bus_uclm_data()
    data2 = load_busi_data()
    return data1 + data2


def read_image(img_path, gray=False):
    img = _pil_read_cv2(img_path, gray=gray)
    if img is None:
        return None
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img, size=IMG_SIZE, normalize=True):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, size)
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img


def _group_split_by_patients(records, test_size=0.2, label_key="label_detection"):
    """
    按患者ID分组划分，防止患者级数据泄漏。
    同时按(source, label)分层，保证各子集比例一致。
    返回: train_records, test_records
    """
    rng = np.random.RandomState(RANDOM_SEED)
    # 按 (source, label, patient_id) 分组
    groups = defaultdict(list)
    for r in records:
        pid = r.get("patient_id", r["img_path"])
        lbl = r[label_key] if r[label_key] is not None else "none"
        key = (r["source"], str(lbl))
        groups[key].append(pid)
    
    train_patients = set()
    test_patients = set()
    
    for key, patient_ids in groups.items():
        # 去重，同一患者可能有多张图
        unique_patients = list(set(patient_ids))
        rng.shuffle(unique_patients)
        n_test = max(1, int(len(unique_patients) * test_size))
        test_patients.update(unique_patients[:n_test])
        train_patients.update(unique_patients[n_test:])
    
    train_records = [r for r in records if r.get("patient_id", r["img_path"]) in train_patients]
    test_records = [r for r in records if r.get("patient_id", r["img_path"]) in test_patients]
    return train_records, test_records


def split_detection_data(records, test_size=0.2):
    """按患者分组划分检测任务数据，同时按来源分层，防止患者级泄漏"""
    train_rec, test_rec = _group_split_by_patients(records, test_size, "label_detection")
    X_train = [r["img_path"] for r in train_rec]
    y_train = [r["label_detection"] for r in train_rec]
    X_test = [r["img_path"] for r in test_rec]
    y_test = [r["label_detection"] for r in test_rec]
    return X_train, X_test, y_train, y_test


def split_classify_data(records, test_size=0.2, unlabeled_ratio=0.6):
    """
    分类任务划分：按患者分组
    1. 先从所有有标签样本中按患者划出一部分作为"无标签"
    2. 剩余有标签样本再按患者划分为train/test
    """
    cls_records = [r for r in records if r["label_classify"] is not None]
    
    # 第一步：划出无标签患者（模拟半监督场景）
    labeled_rec, unlabeled_rec = _group_split_by_patients(
        cls_records, test_size=unlabeled_ratio, label_key="label_classify"
    )
    
    # 第二步：剩余有标签样本划分为train/test
    train_rec, test_rec = _group_split_by_patients(
        labeled_rec, test_size=test_size, label_key="label_classify"
    )
    
    X_train = [r["img_path"] for r in train_rec]
    y_train = [r["label_classify"] for r in train_rec]
    mask_train = [r["mask_path"] for r in train_rec]
    
    X_test = [r["img_path"] for r in test_rec]
    y_test = [r["label_classify"] for r in test_rec]
    mask_test = [r["mask_path"] for r in test_rec]
    
    X_unlabeled = [r["img_path"] for r in unlabeled_rec]
    mask_unlabeled = [r["mask_path"] for r in unlabeled_rec]
    
    return {
        "X_train": X_train, "y_train": y_train, "mask_train": mask_train,
        "X_test": X_test, "y_test": y_test, "mask_test": mask_test,
        "X_unlabeled": X_unlabeled, "mask_unlabeled": mask_unlabeled
    }


if __name__ == "__main__":
    records = load_all_data()
    print(f"Total records: {len(records)}")
    from collections import Counter
    det_labels = [r["label_detection"] for r in records]
    print(f"Detection distribution: {Counter(det_labels)}")
    cls_labels = [r["label_classify"] for r in records if r["label_classify"] is not None]
    print(f"Classify distribution: {Counter(cls_labels)}")
    split = split_classify_data(records)
    print(f"Labeled train: {len(split['X_train'])}, test: {len(split['X_test'])}, unlabeled: {len(split['X_unlabeled'])}")