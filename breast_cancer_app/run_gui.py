"""
永不闪退的GUI启动脚本
双击 run_gui.py 或 命令行执行 都会保留错误信息
"""
import sys
import os
import traceback
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

LOG_FILE = os.path.join(SCRIPT_DIR, "gui_startup_log.txt")

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass

log("=" * 60)
log("GUI启动开始 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
log("Python 可执行文件: " + sys.executable)
log("Python 版本: " + sys.version)
log("工作目录: " + os.getcwd())
log("脚本目录: " + SCRIPT_DIR)
log("=" * 60)

try:
    log("\n[1/5] 导入 Tkinter...")
    import tkinter as tk
    from tkinter import messagebox
    log("  ✓ Tkinter OK")
except Exception as e:
    log(f"  ✗ Tkinter 失败: {e}")
    traceback.print_exc()
    log("\n请按回车键退出...")
    try: input()
    except: pass
    sys.exit(1)

def critical_error_popup(title, body):
    try:
        root = tk.Tk()
        root.title(title)
        root.geometry("820x560")
        root.configure(bg="#fff5f5")
        tk.Label(root, text="❌ " + title, font=("微软雅黑", 15, "bold"),
                 fg="#c62828", bg="#fff5f5").pack(pady=16)
        txt = tk.Text(root, wrap="word", font=("Consolas", 10), height=20)
        txt.pack(padx=20, pady=4, fill="both", expand=True)
        txt.insert("1.0", body)
        txt.configure(state="disabled")
        tk.Label(root, text="完整日志已保存到: gui_startup_log.txt",
                 font=("微软雅黑", 9), fg="#d84315", bg="#fff5f5").pack(pady=4)
        tk.Button(root, text="关闭", width=14, command=root.destroy,
                  bg="#e53935", fg="white").pack(pady=14)
        root.mainloop()
    except Exception:
        pass

try:
    log("\n[2/5] 导入核心依赖 (numpy/PIL)...")
    try:
        import numpy as np
        log("  ✓ numpy " + str(getattr(np, "__version__", "OK")))
    except Exception as e:
        log(f"  ✗ numpy 失败: {e}")
        critical_error_popup("numpy 未安装", traceback.format_exc())
        raise

    try:
        from PIL import Image, ImageTk
        log("  ✓ Pillow OK")
    except Exception as e:
        log(f"  ⚠ Pillow 失败: {e}")

    log("\n[3/5] 导入项目模块...")
    modules_status = {}
    for name in ("config", "data_loader", "feature_extractor",
                 "classification_model", "visualization", "detection_model"):
        try:
            __import__(name)
            modules_status[name] = "OK"
            log(f"  ✓ {name}")
        except Exception as e:
            tb = traceback.format_exc()
            modules_status[name] = f"FAIL: {e}"
            log(f"  ✗ {name}: {e}")
            log("    完整堆栈如下:")
            for line in tb.strip().splitlines():
                log("    | " + line)
            log("    -----")
            # 不raise，继续导入，看看后面的

    log("\n[4/5] 导入 gui_app 模块...")
    try:
        import gui_app
        log("  ✓ gui_app OK")
    except Exception as e:
        tb = traceback.format_exc()
        log(f"  ✗ gui_app 导入失败: {e}")
        log(tb)
        critical_error_popup("gui_app 模块导入失败",
                             "错误:\n" + repr(e) + "\n\n堆栈:\n" + tb)
        raise

    log("\n[5/5] 启动 GUI 主窗口...")
    log("  (图形界面应该弹出在桌面上了)")
    try:
        app = gui_app.BreastCancerApp()
        log("  ✓ BreastCancerApp 实例创建成功，进入 mainloop()")
        app.mainloop()
        log("  ✓ GUI 正常退出")
    except Exception as e:
        tb = traceback.format_exc()
        log(f"  ✗ GUI 启动失败: {e}")
        log(tb)
        critical_error_popup("GUI 启动失败",
                             "错误:\n" + repr(e) + "\n\n堆栈:\n" + tb)
        raise

except Exception as e:
    log("\n" + "=" * 60)
    log("启动过程发生异常: " + repr(e))
    log("=" * 60)
    traceback.print_exc()
    log("\n完整日志请查看: " + LOG_FILE)
    log("\n请按回车键退出...")
    try:
        input()
    except Exception:
        pass
    sys.exit(1)

log("\n✅ 程序结束")
log("日志文件: " + LOG_FILE)