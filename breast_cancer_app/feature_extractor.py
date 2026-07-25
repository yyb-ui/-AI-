import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments_hu, regionprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.stats import skew, kurtosis
import pandas as pd
from data_loader import read_image, preprocess_image, _pil_read_cv2
from config import FEATURE_NAMES


def auto_segment_lesion(image_gray):
    if image_gray is None:
        return None
    try:
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image_gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask
    except Exception:
        return None


def load_mask(mask_path, shape=None):
    if mask_path and os.path.exists(mask_path):
        mask = _pil_read_cv2(mask_path, gray=True)
        if mask is not None:
            if shape is not None:
                mask = cv2.resize(mask, (shape[1], shape[0]))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return mask
    return None


def extract_shape_features(mask, pixel_area=1.0):
    features = {}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(cnt), 1)
    perimeter = max(cv2.arcLength(cnt, True), 1e-6)
    features["area_ratio"] = area / max(mask.shape[0] * mask.shape[1], 1)
    features["perimeter"] = perimeter
    features["circularity"] = (4 * np.pi * area) / (perimeter ** 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    features["major_axis"] = max(w, h)
    features["minor_axis"] = min(w, h)
    features["eccentricity"] = 0 if max(w, h) == 0 else np.sqrt(1 - (min(w, h) / max(w, h)) ** 2)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1)
    features["solidity"] = area / hull_area
    equi_diameter = np.sqrt(4 * area / np.pi)
    features["equivalent_diameter"] = equi_diameter
    features["compactness"] = (perimeter ** 2) / (4 * np.pi * area)
    hu = moments_hu(mask.astype(np.float64))
    for i in range(7):
        features[f"hu_moment_{i+1}"] = -np.sign(hu[i]) * np.log10(np.abs(hu[i]) + 1e-10)
    return features


def extract_intensity_features(image_gray, mask=None):
    if mask is not None and np.sum(mask > 0) > 10:
        roi = image_gray[mask > 0]
    else:
        roi = image_gray.flatten()
    if len(roi) == 0:
        roi = image_gray.flatten()
    features = {}
    features["mean_intensity"] = np.mean(roi)
    features["std_intensity"] = np.std(roi)
    features["skewness"] = skew(roi) if np.std(roi) > 1e-6 else 0
    features["kurtosis"] = kurtosis(roi) if np.std(roi) > 1e-6 else 0
    hist, _ = np.histogram(roi, bins=256, range=(0, 255), density=True)
    hist = hist + 1e-10
    features["entropy"] = -np.sum(hist * np.log2(hist))
    features["texture_uniformity"] = np.sum(hist ** 2)
    return features


def extract_texture_features(image_gray, mask=None):
    features = {}
    if mask is not None and np.sum(mask > 0) > 10:
        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = max(0, y_indices.min() - 5), min(image_gray.shape[0], y_indices.max() + 5)
        x_min, x_max = max(0, x_indices.min() - 5), min(image_gray.shape[1], x_indices.max() + 5)
        roi = image_gray[y_min:y_max, x_min:x_max]
    else:
        roi = image_gray
    if roi.size == 0:
        roi = image_gray
    h, w = roi.shape
    if h < 5 or w < 5:
        roi = cv2.resize(image_gray, (100, 100))
    roi_resized = cv2.resize(roi, (100, 100))
    # 修复uint8溢出bug：先转float再归一化，避免roi_resized*255时uint8回绕
    roi_float = roi_resized.astype(np.float32)
    roi_max = roi_float.max()
    roi_uint8 = (roi_float * 255.0 / roi_max).astype(np.uint8) if roi_max > 0 else roi_resized.astype(np.uint8)
    glcm = graycomatrix(roi_uint8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)
        features[prop] = np.mean(vals)
    lbp = local_binary_pattern(roi_uint8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    # 原命名fractal_dimension不准确，实际是LBP直方图的香农熵
    features["lbp_entropy"] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)) / np.log2(max(lbp.size, 2))
    return features


def extract_all_features(img_path, mask_path=None):
    img = read_image(img_path, gray=False)
    if img is None:
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    mask = load_mask(mask_path, shape=img_gray.shape)
    auto_segmented = False
    if mask is None:
        mask = auto_segment_lesion(img_gray)
        auto_segmented = mask is not None
    if mask is None:
        # 自动分割失败，退化为全图分析（形状特征不可靠，仅用强度和纹理）
        mask = np.ones_like(img_gray) * 255
    shape_feats = extract_shape_features(mask)
    if shape_feats is None:
        # 形状提取失败时，用全图尺寸估算默认形状特征，避免全0导致模型偏差
        h, w = img_gray.shape[:2]
        area = h * w
        perimeter = 2 * (h + w)
        shape_feats = {
            "area_ratio": 1.0, "perimeter": float(perimeter),
            "circularity": (4 * np.pi * area) / (perimeter ** 2),
            "major_axis": float(max(w, h)), "minor_axis": float(min(w, h)),
            "eccentricity": np.sqrt(1 - (min(w, h) / max(w, h)) ** 2),
            "solidity": 1.0, "equivalent_diameter": np.sqrt(4 * area / np.pi),
            "compactness": (perimeter ** 2) / (4 * np.pi * area),
        }
        for i in range(7):
            shape_feats[f"hu_moment_{i+1}"] = 0.0
    intensity_feats = extract_intensity_features(img_gray, mask)
    texture_feats = extract_texture_features(img_gray, mask)
    all_feats = {**shape_feats, **intensity_feats, **texture_feats}
    vec = []
    for name in FEATURE_NAMES:
        val = all_feats.get(name, 0)
        if not isinstance(val, (int, float)):
            val = float(val)
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        vec.append(val)
    return np.array(vec, dtype=np.float32)


def batch_extract_features(img_paths, mask_paths=None, progress_callback=None):
    if mask_paths is None:
        mask_paths = [None] * len(img_paths)
    features = []
    valid_idx = []
    total = len(img_paths)
    for i, (p, m) in enumerate(zip(img_paths, mask_paths)):
        try:
            feat = extract_all_features(p, m)
            if feat is not None:
                features.append(feat)
                valid_idx.append(i)
        except Exception as e:
            pass
        if progress_callback and i % 20 == 0:
            progress_callback(i, total)
    if progress_callback:
        progress_callback(total, total)
    return np.array(features, dtype=np.float32), valid_idx


if __name__ == "__main__":
    from data_loader import load_all_data
    records = load_all_data()
    r = records[10]
    feat = extract_all_features(r["img_path"], r["mask_path"])
    print(f"Feature dimension: {len(feat)}")
    print("Feature names:")
    for n, v in zip(FEATURE_NAMES, feat):
        print(f"  {n}: {v:.4f}")