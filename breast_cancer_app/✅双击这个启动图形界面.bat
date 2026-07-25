@echo off
chcp 65001 >nul
title ⚡⚡⚡ 一键启动图形界面（必用 radiomics 环境）
cd /d "%~dp0"

rem ===== 必用 radiomics Python 3.11 =====
set "RAD_PY=C:\Users\lenovo\anaconda3\envs\radiomics\python.exe"
set "LOG=%~dp0LAST_LAUNCH_LOG.txt"

echo [%date% %time%] LAUNCH START > "%LOG%"

if not exist "%RAD_PY%" (
    echo ========================================================
    echo   ❌ 未找到 radiomics 环境！
    echo   预期路径: %RAD_PY%
    echo.
    echo   请先确认 anaconda 里已创建 radiomics 环境
    echo ========================================================
    echo Radiomics env MISSING >> "%LOG%"
    pause
    exit /b 1
)

echo ========================================================
echo   ✅ 使用 radiomics 环境 Python 3.11
echo   解释器: %RAD_PY%
echo   脚本: %~dp0run_gui.py
echo.
echo   🚀 图形界面将在 2-10 秒后弹出...
echo ========================================================

echo Use python: %RAD_PY% >> "%LOG%"
echo Script: %~dp0run_gui.py >> "%LOG%"
echo. >> "%LOG%"

"%RAD_PY%" "%~dp0run_gui.py"

echo Exit code: %errorlevel% >> "%LOG%"
echo.
echo ========================================================
echo   程序已退出。日志文件: LAST_LAUNCH_LOG.txt
echo   若有问题，请把日志内容/截图发给我
echo ========================================================
pause