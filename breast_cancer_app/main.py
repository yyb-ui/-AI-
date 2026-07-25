import os
import sys
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def check_environment():
    print("=" * 60)
    print("🏥 乳腺癌超声影像智能诊断系统  环境检查")
    print("=" * 60)
    def _safe_version(mod, default="可用"):
        for attr in ("__version__", "version", "VERSION"):
            v = getattr(mod, attr, None)
            if isinstance(v, str) and v:
                return v
            if callable(v):
                try:
                    r = v()
                    if isinstance(r, str) and r:
                        return r
                except Exception:
                    pass
        try:
            from importlib.metadata import version as _v
            return _v(mod.__name__)
        except Exception:
            return default

    errors = []
    try:
        import numpy, pandas, scipy
        print(f"✓ 科学计算: numpy {_safe_version(numpy)}, pandas {_safe_version(pandas)}, scipy {_safe_version(scipy)}")
    except ImportError as e:
        errors.append(f"numpy/pandas/scipy 未安装: {e}")
    try:
        import sklearn, skimage
        print(f"✓ 机器学习: sklearn {_safe_version(sklearn)}, scikit-image OK")
    except ImportError as e:
        errors.append(f"scikit-learn/scikit-image 未安装: {e}")
    try:
        import tensorflow as tf
        tf_ver = _safe_version(tf, "兼容版本")
        print(f"✓ 深度学习: TensorFlow {tf_ver}")
        try:
            gpus = tf.config.list_physical_devices("GPU")
            print(f"  GPU可用数量: {len(gpus)} {'(启用GPU加速)' if gpus else '(使用CPU)'}")
        except Exception:
            try:
                from tensorflow.python.client import device_lib
                devs = device_lib.list_local_devices()
                gpus = [d for d in devs if d.device_type == "GPU"]
                print(f"  GPU可用数量: {len(gpus)} {'(启用GPU加速)' if gpus else '(使用CPU)'}")
            except Exception:
                print(f"  GPU信息: 未检测到(CUDA环境可能未配置)")
    except ImportError as e:
        errors.append(f"TensorFlow 未安装: {e}")
    try:
        import cv2
        print(f"✓ 图像处理: OpenCV {_safe_version(cv2)}")
    except ImportError as e:
        errors.append(f"OpenCV 未安装: {e}")
    try:
        import shap
        print(f"✓ 可解释性: SHAP {_safe_version(shap)}")
    except ImportError as e:
        print(f"⚠ SHAP 未安装(可解释性功能不可用): {e}")
    try:
        import matplotlib, seaborn
        print(f"✓ 可视化: matplotlib {_safe_version(matplotlib)}, seaborn OK")
    except ImportError as e:
        errors.append(f"matplotlib/seaborn 未安装: {e}")
    try:
        import tkinter
        print(f"✓ GUI框架: Tkinter OK")
    except ImportError as e:
        errors.append(f"Tkinter 不可用: {e}")
    print("-" * 60)
    from config import DATA_DIR
    busi_ok = os.path.exists(DATA_DIR)
    print(f"✓ 数据目录: {'存在' if busi_ok else '不存在'} -> {DATA_DIR}")
    if errors:
        print("❌ 以下错误需修复后运行:")
        for e in errors:
            print(f"  - {e}")
        print("\n安装命令: pip install -r requirements.txt")
        return False
    return True


def train_all_models(progress_callback=None):
    from detection_model import train_detection_model
    from classification_model import train_classification_models
    print("\n🚀 开始训练完整两阶段模型...")
    model, hist1, hist2, _ = train_detection_model(progress_callback=progress_callback, epochs=20)
    result = train_classification_models(progress_callback=progress_callback)
    print("\n✅ 所有模型训练完成！")
    return model, result


def main():
    try:
        ok = check_environment()
    except Exception as e:
        print(f"环境检查异常(可忽略): {e}")
        import traceback; traceback.print_exc()
        ok = True
    if not ok:
        print("\n⚠️ 环境未完全就绪，仍尝试启动GUI...(按Ctrl+C退出)")
    print("\n" + "=" * 60)
    print("📋 快速开始菜单:")
    print("=" * 60)
    print("  1) 直接启动图形界面 (推荐)")
    print("  2) 先训练两阶段模型，再启动界面")
    print("  3) 仅训练模型(后台)")
    print("  q) 退出")
    try:
        choice = input("\n请输入选项 [1]: ").strip() or "1"
    except Exception:
        choice = "1"
    if choice.lower() == "q":
        return
    if choice in ["2", "3"]:
        print("\n⚠️ 模型训练将需要较长时间（5-30分钟，取决于硬件）")
        try:
            confirm = input("确认开始训练？(y/N): ").strip().lower()
        except Exception:
            confirm = "n"
        if confirm == "y":
            try:
                train_all_models()
            except Exception as e:
                print(f"❌ 训练失败: {e}")
                import traceback; traceback.print_exc()
        else:
            print("已取消训练。")
    if choice in ["1", "2"]:
        print("\n启动图形界面中...")
        try:
            from gui_app import main as gui_main
            gui_main()
        except Exception as e:
            print(f"❌ GUI启动失败: {e}")
            import traceback; traceback.print_exc()
            print("\n请双击运行: 🔥调试启动_不闪退.bat 获取详细错误信息")
            try: input("\n按回车键退出...")
            except: pass
    elif choice == "3":
        print("\n训练完成，程序退出。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户取消。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            print(f"\n❌ 主程序异常: {e}")
            input("按回车键退出...")
        except Exception:
            pass