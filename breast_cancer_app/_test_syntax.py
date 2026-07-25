import sys
import os

# 设置工作目录
work_dir = r"c:\Users\lenovo\Desktop\2026暑期卓工平台\practice\breast_cancer_app"
os.chdir(work_dir)
sys.path.insert(0, work_dir)

print(f"工作目录: {os.getcwd()}")
print(f"文件是否存在: {os.path.exists('gui_app.py')}")

try:
    import py_compile
    py_compile.compile('gui_app.py', doraise=True)
    print("✅ 语法检查通过")
except Exception as e:
    print(f"❌ 语法错误: {e}")
    import traceback
    traceback.print_exc()

try:
    import gui_app
    print("✅ 模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()