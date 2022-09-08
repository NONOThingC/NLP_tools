import os
import sys
# 找到当前文件的决定路径,__file__ 表示当前文件,也就是test.py
file_path = os.path.abspath(__file__)
print(file_path)
# 获取当前文件所在的目录
cur_path = os.path.dirname(file_path)
print(cur_path)
# 获取项目所在路径
project_path = os.path.dirname(cur_path)
print(project_path)
# 把项目路径加入python搜索路径
sys.path.append(project_path)