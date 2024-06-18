#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/14 12:00
# @Author  : 兵
# @email    : 1747193328@qq.com
import os.path
import sys

from ase.io import read, write
from sklearn.model_selection import train_test_split

if os.path.exists("train-0.9.xyz"):
    print("当前目录下已经有train-0.9.xyz文件，将追加到文件，而不是覆盖写入。")
if os.path.exists("test-0.1.xyz"):
    print("当前目录下已经有train-0.9.xyz文件，将追加到文件，而不是覆盖写入。")
atoms = read(sys.argv[1], ":", format="extxyz")
train, test = train_test_split(atoms, test_size=0.1, random_state=88, shuffle=True)
# 这里append=True 考虑可以将多个体系合并下

write("./train-0.9.xyz", train, format='extxyz', append=True)
write("./test-0.1.xyz", test, format='extxyz', append=True)
