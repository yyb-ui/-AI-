import os as _os
_os.environ.setdefault("PYTHONHASHSEED", str(42))
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")   # TF确定性模式(GPU训练时)
_os.environ.setdefault("CUDNN_DETERMINISTIC", "1")    # CuDNN确定性模式(有N卡时)

import numpy as _np
_np.random.seed(42)
import random as _random
_random.seed(42)

# 下面是原来的常规导入（保持你原来的顺序和内容即可）
import os
import numpy as np
import cv2
from data_loader import read_image, preprocess_image, split_detection_data, load_all_data
from config import IMG_SIZE, BATCH_SIZE, EPOCHS_DETECTION, MODEL_DIR, RANDOM_SEED

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, callbacks
    try:
        from tensorflow.keras import backend as K
    except Exception:
        K = None
    try:
        tf.random.set_seed(RANDOM_SEED)
        # 让 Keras 图层面的操作也固定种子
        if K is not None:
            try:
                K.clear_session()
                K.set_image_data_format('channels_last')
            except Exception:
                pass
    except Exception:
        pass
    TF_AVAILABLE = True
except Exception as e:
    tf = None
    layers = None
    models = None
    applications = None
    callbacks = None
    K = None
    TF_AVAILABLE = False
    print(f"⚠ TensorFlow 不可用(检测模块将禁用): {e}")

np.random.seed(RANDOM_SEED)


def build_detection_cnn(input_shape=(224, 224, 3), num_classes=2, use_pretrained=True):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow未安装，无法构建检测模型")
    if use_pretrained:
        base = applications.MobileNetV2(
            input_shape=input_shape, include_top=False,
            weights="imagenet", alpha=0.75
        )
        base.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs)
    else:
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])
    loss = "binary_crossentropy"
    try:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    except Exception:
        opt = tf.optimizers.Adam(learning_rate=1e-3)
    try:
        metrics = ["accuracy",
                   tf.keras.metrics.AUC(name="auc"),
                   tf.keras.metrics.Recall(name="recall"),
                   tf.keras.metrics.Precision(name="precision")]
    except Exception:
        metrics = ["accuracy"]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def data_generator(img_paths, labels, batch_size=BATCH_SIZE, augment=False, class_weight=None):
    n = len(img_paths)
    while True:
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            X, y, sw = [], [], []
            for i in batch_idx:
                img = read_image(img_paths[i])
                if img is None:
                    continue
                img = preprocess_image(img)
                if augment:
                    img = _augment(img)
                X.append(img)
                label_val = int(labels[i])
                y.append(label_val)
                if class_weight is not None:
                    sw.append(float(class_weight.get(label_val, 1.0)))
                else:
                    sw.append(1.0)
            if not X:
                continue
            if class_weight is not None:
                yield np.array(X), np.array(y), np.array(sw)
            else:
                yield np.array(X), np.array(y)


def _augment(img):
    if np.random.random() < 0.5:
        img = img[:, ::-1]
    if np.random.random() < 0.5:
        img = np.rot90(img, k=np.random.choice([1, 2, 3]))
    if np.random.random() < 0.3:
        alpha = 0.8 + np.random.random() * 0.4
        beta = -20 + np.random.random() * 40
        img = np.clip(alpha * img + beta / 255.0, 0, 1)
    return img


def train_detection_model(progress_callback=None, epochs=EPOCHS_DETECTION):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow未安装，无法训练检测模型")
    records = load_all_data()
    # 第一级划分：trainval + test（test仅用于最终评估，不参与训练调参）
    X_trainval, X_test, y_trainval, y_test = split_detection_data(records, test_size=0.2)
    # 第二级划分：train + val（val用于EarlyStopping和Checkpoint选最优）
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y_trainval
    )
    # 计算类别权重，处理样本不平衡
    neg = sum(1 for y in y_train if y == 0)
    pos = sum(1 for y in y_train if y == 1)
    total = neg + pos
    class_weight = {
        0: total / (2 * neg) if neg > 0 else 1.0,
        1: total / (2 * pos) if pos > 0 else 1.0
    }
    if progress_callback:
        progress_callback("正在构建病灶检测模型(MobileNetV2迁移学习)...", 5)
    model = build_detection_cnn(use_pretrained=True)
    train_gen = data_generator(X_train, y_train, augment=True, class_weight=class_weight)
    val_gen = data_generator(X_val, y_val, augment=False, class_weight=class_weight)
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    val_steps = max(1, len(X_val) // BATCH_SIZE)
    save_path = os.path.join(MODEL_DIR, "detection_model.h5")
    def _safe_callbacks(monitor_metric="val_auc"):
        cbs = []
        try:
            cbs.append(callbacks.ModelCheckpoint(save_path, monitor=monitor_metric,
                                                  save_best_only=True, mode="max", verbose=0))
        except Exception:
            pass
        try:
            cbs.append(callbacks.EarlyStopping(monitor=monitor_metric, patience=5, mode="max",
                                                restore_best_weights=True))
        except Exception:
            pass
        try:
            cbs.append(callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                    patience=3, min_lr=1e-6, verbose=0))
        except Exception:
            pass
        return cbs
    cb_list = _safe_callbacks()
    if progress_callback:
        try:
            class ProgCb(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pct = 10 + int((epoch + 1) / epochs * 80)
                    logs = logs or {}
                    msg = f"Epoch {epoch+1}/{epochs} | AUC(val)={logs.get('val_auc', logs.get('val_accuracy', 0)):.4f}"
                    progress_callback(msg, pct)
            cb_list.append(ProgCb())
        except Exception:
            pass
    if progress_callback:
        progress_callback(f"开始训练 | 训练集:{len(X_train)} 验证集:{len(X_val)} 测试集:{len(X_test)}", 10)
    history = model.fit(
        train_gen, steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, validation_steps=val_steps,
        epochs=epochs, callbacks=cb_list, verbose=1,
    )
    if progress_callback:
        progress_callback("检测模型训练完成，启动第二阶段微调...", 92)
    try:
        model.layers[1].trainable = True
        for layer in model.layers[1].layers[:80]:
            layer.trainable = False
    except Exception:
        pass
    try:
        opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    except Exception:
        opt = tf.optimizers.Adam(learning_rate=2e-5)
    try:
        ft_metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"),
                      tf.keras.metrics.Recall(name="recall"),
                      tf.keras.metrics.Precision(name="precision")]
    except Exception:
        ft_metrics = ["accuracy"]
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=ft_metrics)
    history2 = model.fit(
        train_gen, steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, validation_steps=val_steps,
        epochs=max(5, epochs // 2),
        callbacks=_safe_callbacks(), verbose=1,
    )
    try:
        model.save(save_path)
    except Exception:
        pass
    if progress_callback:
        progress_callback("病灶检测模型训练完成！", 100)
    return model, history, history2, (X_test, y_test)


def load_detection_model():
    if not TF_AVAILABLE:
        return None
    path = os.path.join(MODEL_DIR, "detection_model.h5")
    if not os.path.exists(path):
        return None
    try:
        try:
            model = tf.keras.models.load_model(path, compile=False)
        except Exception:
            model = models.load_model(path, compile=False)
        return model
    except Exception as e:
        print(f"Load model error: {e}")
        return None


def predict_detection(model, img_path):
    if model is None:
        return None, None
    img = read_image(img_path)
    if img is None:
        return None, None
    x = preprocess_image(img)
    x = np.expand_dims(x, axis=0)
    try:
        preds = model.predict(x, verbose=0)[0]
    except Exception:
        preds = model(x, training=False).numpy()[0]
    if np.ndim(preds) == 0:
        prob = float(preds)
    else:
        prob = float(preds[-1]) if len(preds) <= 2 else float(np.argmax(preds))
    label = 1 if prob >= 0.5 else 0
    return label, prob


def _find_conv_layer(container_model, prefer_last=True):
    candidates = []
    try:
        if hasattr(container_model, "layers"):
            for idx, layer in enumerate(container_model.layers):
                lname = getattr(layer, "name", "").lower()
                ltype = type(layer).__name__
                hit = False
                if hasattr(layer, "output_shape"):
                    shp = layer.output_shape
                    if isinstance(shp, (list, tuple)) and len(shp) >= 4 and shp[1] is not None and shp[2] is not None and shp[3] is not None:
                        if shp[1] > 3 and shp[2] > 3:
                            hit = True
                if "conv" in lname or "add" in lname or "out_relu" in lname or "block" in lname or "bn" in lname:
                    hit = True
                if "Conv" in ltype or "Add" in ltype:
                    hit = True
                if hit:
                    candidates.append((idx, layer))
                if hasattr(layer, "layers"):
                    sub = _find_conv_layer(layer, prefer_last=prefer_last)
                    for s in sub:
                        candidates.append((-1, s[1]))
    except Exception:
        pass
    return candidates


def _fallback_saliency_heatmap(img_np, orig_h, orig_w):
    try:
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()
        if gray.max() <= 1.0:
            gray_u = (gray * 255).astype(np.uint8)
        else:
            gray_u = gray.astype(np.uint8)
        gx = cv2.Sobel(gray_u, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_u, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        if mag.max() > 1e-8:
            mag /= mag.max()
        blur = cv2.GaussianBlur(mag, (0, 0), sigmaX=15, sigmaY=15)
        if blur.max() > 1e-8:
            blur = (blur - blur.min()) / (blur.max() - blur.min() + 1e-9)
        th = np.percentile(blur, 40)
        heat = np.where(blur > th, blur, 0.0)
        if heat.max() > 1e-8:
            heat = heat / (heat.max() + 1e-9)
        heatmap = cv2.resize(heat.astype(np.float32), (orig_w, orig_h))
        return heatmap
    except Exception as fe:
        print(f"Fallback heatmap err: {fe}")
        return np.zeros((orig_h, orig_w), dtype=np.float32)


def compute_grad_cam(model, img_path, layer_name=None):
    if model is None:
        return None, None
    img = read_image(img_path)
    if img is None:
        return None, None
    orig_h, orig_w = img.shape[:2]
    x = preprocess_image(img)
    x_tensor = np.expand_dims(x, axis=0)
    heatmap = None
    if TF_AVAILABLE:
        try:
            target_layer = None
            base_model_for_grad = None
            if layer_name is not None:
                try:
                    target_layer = model.get_layer(layer_name)
                    base_model_for_grad = model
                except Exception:
                    for sub in model.layers:
                        if hasattr(sub, "get_layer"):
                            try:
                                target_layer = sub.get_layer(layer_name)
                                base_model_for_grad = sub
                                break
                            except Exception:
                                continue
            if target_layer is None:
                cands = _find_conv_layer(model, prefer_last=True)
                if cands:
                    target_layer = cands[-1][1]
                if hasattr(model.layers[1], "layers") if len(model.layers) > 1 else False:
                    base_model_for_grad = model.layers[1]
                    if not cands:
                        sub_cands = _find_conv_layer(base_model_for_grad, prefer_last=True)
                        if sub_cands:
                            target_layer = sub_cands[-1][1]
                else:
                    base_model_for_grad = model
            if target_layer is None:
                if len(model.layers) >= 3:
                    target_layer = model.layers[-3]
                    base_model_for_grad = model
            if target_layer is not None and base_model_for_grad is not None:
                connected = False
                try:
                    grad_model = models.Model(
                        inputs=[base_model_for_grad.input],
                        outputs=[target_layer.output, base_model_for_grad.output]
                    )
                    connected = True
                except Exception as e1:
                    print(f"Grad-CAM nested model route failed: {e1}")
                    try:
                        grad_model = models.Model(
                            inputs=[model.input],
                            outputs=[target_layer.output, model.output]
                        )
                        base_model_for_grad = model
                        connected = True
                    except Exception as e2:
                        print(f"Grad-CAM direct route failed: {e2}")
                        try:
                            last_conv = None
                            for lyr in reversed(base_model_for_grad.layers):
                                if hasattr(lyr, "output_shape"):
                                    shp = lyr.output_shape
                                    if isinstance(shp, tuple) and len(shp) >= 4:
                                        last_conv = lyr
                                        break
                            if last_conv is not None:
                                grad_model = models.Model(
                                    inputs=[base_model_for_grad.input],
                                    outputs=[last_conv.output, base_model_for_grad.output]
                                )
                                target_layer = last_conv
                                connected = True
                        except Exception as e3:
                            print(f"Grad-CAM last conv route failed: {e3}")
                if connected:
                    try:
                        x_tf = tf.convert_to_tensor(x_tensor, dtype=tf.float32)
                        with tf.GradientTape(persistent=False) as tape:
                            tape.watch(x_tf)
                            conv_outputs, predictions = grad_model(x_tf)
                            preds_tensor = tf.convert_to_tensor(predictions)
                            if preds_tensor.shape.ndims >= 2 and preds_tensor.shape[-1] <= 1:
                                loss = preds_tensor[:, 0]
                            elif preds_tensor.shape.ndims >= 2:
                                loss = preds_tensor[:, -1]
                            else:
                                loss = preds_tensor
                        grads = tape.gradient(loss, conv_outputs)
                        if grads is not None:
                            g_np = grads.numpy() if hasattr(grads, "numpy") else np.array(grads)
                            if not np.any(np.isnan(g_np)) and np.max(np.abs(g_np)) > 1e-10:
                                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                                conv_np = conv_outputs[0].numpy() if hasattr(conv_outputs, "numpy") else np.array(conv_outputs[0])
                                pooled_np = pooled_grads.numpy() if hasattr(pooled_grads, "numpy") else np.array(pooled_grads)
                                heatmap_calc = conv_np @ pooled_np[..., np.newaxis]
                                heatmap_calc = np.squeeze(heatmap_calc)
                                heatmap_calc = np.maximum(heatmap_calc, 0.0)
                                if heatmap_calc.max() > 1e-8:
                                    heatmap_calc = heatmap_calc / (heatmap_calc.max() + 1e-9)
                                heatmap_calc = cv2.resize(heatmap_calc.astype(np.float32), (orig_w, orig_h))
                                heatmap = heatmap_calc
                                print("✅ Grad-CAM (标准TF梯度版) 生成成功")
                    except Exception as ge:
                        print(f"Grad-CAM gradient tape failed: {ge}")
            if heatmap is None and TF_AVAILABLE:
                try:
                    print("Grad-CAM 降级: 使用Gradient-based Saliency...")
                    x_tf2 = tf.convert_to_tensor(x_tensor, dtype=tf.float32)
                    with tf.GradientTape() as tape2:
                        tape2.watch(x_tf2)
                        preds = model(x_tf2, training=False)
                        if hasattr(preds, "shape") and preds.shape.ndims >= 2 and preds.shape[-1] <= 1:
                            score = preds[:, 0]
                        elif hasattr(preds, "shape") and preds.shape.ndims >= 2:
                            score = preds[:, -1]
                        else:
                            score = preds
                    sal = tape2.gradient(score, x_tf2)
                    if sal is not None:
                        sal_np = sal.numpy()[0] if hasattr(sal, "numpy") else np.array(sal)[0]
                        sal_abs = np.max(np.abs(sal_np), axis=-1)
                        if sal_abs.max() > 1e-8:
                            sal_abs = (sal_abs - sal_abs.min()) / (sal_abs.max() - sal_abs.min() + 1e-9)
                        sal_blur = cv2.GaussianBlur(sal_abs, (0, 0), sigmaX=8)
                        if sal_blur.max() > 1e-8:
                            sal_blur = (sal_blur - sal_blur.min()) / (sal_blur.max() - sal_blur.min() + 1e-9)
                        heatmap = cv2.resize(sal_blur.astype(np.float32), (orig_w, orig_h))
                        print("✅ Grad-CAM 降级 (Gradient Saliency) 成功")
                except Exception as se:
                    print(f"Saliency fallback failed: {se}")
        except Exception as e:
            print(f"Grad-CAM TF pipeline failed: {e}")
    if heatmap is None or (np.sum(heatmap) < 1e-6):
        print("ℹ️ Grad-CAM 使用最终降级方案 (OpenCV显著性检测)")
        heatmap = _fallback_saliency_heatmap(x, orig_h, orig_w)
    try:
        hm_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
        heatmap_jet = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
        img_f = img.copy()
        if img_f.max() <= 1.0:
            img_f = (img_f * 255.0).astype(np.uint8)
        else:
            img_f = img_f.astype(np.uint8)
        if len(img_f.shape) == 2:
            img_f = cv2.cvtColor(img_f, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(img_f, 0.58, heatmap_jet, 0.42, 0)
        return heatmap, overlay
    except Exception as oe:
        print(f"Heatmap overlay render failed: {oe}")
        return None, None


if __name__ == "__main__":
    model, h1, h2, (xt, yt) = train_detection_model()
    print("Detection model ready!")