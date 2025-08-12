# tests/conftest.py

import sys
import os

# --- 核心逻辑 ---
# 这段代码的唯一目的，是在 Pytest 运行任何测试之前，
# 动态地将我们项目的根目录（即 pyspark_to_snowpark_poc/）
# 添加到 Python 解释器查找模块的路径列表（sys.path）中。

# 1. `os.path.dirname(__file__)`
#    获取当前文件 (conftest.py) 所在的目录的路径。
#    结果: '.../pyspark_to_snowpark_poc/tests'

# 2. `os.path.join(..., '..')`
#    将上一步得到的路径与 '..' (代表上一级目录) 结合起来。
#    结果: '.../pyspark_to_snowpark_poc'

# 3. `os.path.abspath(...)`
#    获取这个路径的绝对、完整表示，以避免任何相对路径问题。
#    结果: '/path/to/your/project/pyspark_to_snowpark_poc'

# 4. `sys.path.insert(0, ...)`
#    将这个项目的根目录路径，插入到 Python 搜索路径列表的最前面。
#    插入到最前面 (索引 0) 可以确保 Python 会优先在这个目录里查找模块，
#    避免与系统中可能存在的同名库冲突。
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- 无需其他代码 ---
# Pytest 会自动发现并执行这个文件。
# 执行完毕后，在任何测试文件中，类似 `from agents.code_analyzer import CodeAnalyzer`
# 这样的导入语句就能正常工作了。