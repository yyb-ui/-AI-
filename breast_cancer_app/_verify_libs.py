"""校验 radiomics 环境里的项目依赖"""
import sys
libs = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("tensorflow", "tensorflow"),
    ("tkinter", "tkinter(内置)"),
]
ok = True
for imp, pkg in libs:
    try:
        m = __import__(imp)
        v = getattr(m, "__version__", "OK")
        print(f"  ✓ {pkg:20s}  {v}")
    except Exception as e:
        ok = False
        print(f"  ✗ {pkg:20s}  未安装: {e}")

try:
    print("\n项目模块测试:")
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for name in ("config", "data_loader", "feature_extractor", "classification_model", "visualization"):
        try:
            __import__(name)
            print(f"  ✓ {name}")
        except Exception as e:
            import traceback
            ok = False
            print(f"  ✗ {name}: {e}")
            print("    " + "\n    ".join(traceback.format_exc().strip().splitlines()[-3:]))
except Exception as e:
    print(f"测试异常: {e}")
    ok = False

print("\n" + ("✅ 所有库/模块均可正常使用！" if ok else "❌ 有缺失或失败项，请查看上方内容"))
sys.exit(0 if ok else 1)