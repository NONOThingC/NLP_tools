# coding = utf-8
import re


num = 'a￥1aB23Cqqq$我.04'
print("原字符串： ", num)
# 字符串只保留中文
num1 = re.sub(u"([^\u4e00-\u9fa5])", "", num)
print("字符串只保留中文： ", num1)
# 字符串只保留英文
num2 = re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", num)
print("字符串只保留英文： ", num2)
# 字符串只保留数字
num3 = re.sub(u"([^\u0030-\u0039])", "", num)
print("字符串只保留数字： ", num3)
num4 = re.sub("\D", "", num)
print("字符串只保留数字： ", num4)
# 字符串保留数字.和￥
num5 = re.sub(u"([^\u0030-\u0039\u002e\uffe5])", "", num)
print("字符串保留数字.和￥： ", num5)
# 字符串只保留英文和数字
num6 = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", num)
print("字符串只保留英文和数字： ", num6)
