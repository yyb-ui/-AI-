@echo off
chcp 65001 >nul
title 🧰 radiomics环境 - 一键补装项目依赖（版本锁定）
cd /d "%~dp0"

set "RAD_PY=C:\Users\lenovo\anaconda3\envs\radiomics\python.exe"
set "MIRROR=-i https://pypi.tuna.tsinghua.edu.cn/simple"

if not exist "%RAD_PY%" (
    echo ❌ 找不到 radiomics 环境: %RAD_PY%
    pause
    exit /b 1
)

echo ============================================================
echo   🧰 radiomics环境 补装项目依赖（版本锁定 + 冲突规避）
echo   Python: %RAD_PY%
echo.
echo   核心策略: numpy 固定为 1.26.4 (兼容 TensorFlow + OpenCV + SHAP)
echo ============================================================
echo.

echo [0/9] 卸载冲突版 numpy / scipy (强制清理)...
"%RAD_PY%" -m pip uninstall -y numpy scipy scikit-learn scikit-image
echo.

echo [1/9] 安装锁定版 numpy=1.26.4 ...
"%RAD_PY%" -m pip install --no-cache-dir "numpy==1.26.4" %MIRROR%
if errorlevel 1 (
    echo   numpy 安装失败！尝试加 --force-reinstall ...
    "%RAD_PY%" -m pip install --force-reinstall --no-cache-dir "numpy==1.26.4" %MIRROR%
)
echo.

echo [2/9] 安装 scipy pandas (与 numpy 1.26 匹配) ...
"%RAD_PY%" -m pip install "scipy>=1.9,<1.14" "pandas>=1.5,<2.3" %MIRROR%
echo.

echo [3/9] 机器学习: scikit-learn scikit-image
"%RAD_PY%" -m pip install "scikit-learn>=1.2,<1.6" "scikit-image>=0.20,<0.25" %MIRROR%
echo.

echo [4/9] 图像处理: opencv-python Pillow (指定兼容opencv)
"%RAD_PY%" -m pip install --no-deps "opencv-python>=4.7,<4.11" %MIRROR%
"%RAD_PY%" -m pip install "Pillow>=9.5,<11" %MIRROR%
echo.

echo [5/9] 可视化: matplotlib seaborn
"%RAD_PY%" -m pip install "matplotlib>=3.7,<3.10" "seaborn>=0.12,<0.14" %MIRROR%
echo.

echo [6/9] 集成学习: xgboost lightgbm
"%RAD_PY%" -m pip install "xgboost>=1.7,<2.2" "lightgbm>=3.3,<4.6" %MIRROR%
echo.

echo [7/9] 可解释性: shap 0.45 (需 numpy>=1.21 且 <2)
"%RAD_PY%" -m pip install "shap>=0.42,<0.47" %MIRROR%
echo.

echo [8/9] 深度学习: tensorflow (兼容numpy 1.26)
"%RAD_PY%" -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo   未检测到 TensorFlow，安装 tensorflow-intel 2.16 (已兼容 numpy 1.26)...
    "%RAD_PY%" -m pip install "tensorflow-intel==2.16.2" "keras==3.6.0" %MIRROR%
) else (
    echo   TensorFlow 已安装，跳过
)
echo.

echo [9/9] 最终校验所有库导入...
"%RAD_PY%" "%~dp0_verify_libs.py"

echo.
echo ============================================================
echo   ✅ 完成！如无错误，可启动图形界面
echo ============================================================
pause