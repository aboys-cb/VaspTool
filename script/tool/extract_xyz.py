#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 14:32
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
等间距抽取多少个结构在xyz文件中
"""
import sys

from ase.io import read, write

file_path = sys.argv[1]

atoms = read(file_path, index=":", format="extxyz")
if len(sys.argv) == 3:
    num = int(sys.argv[2])
else:
    num = 50

extract = atoms[::num]
print(len(extract))
# 这里将抽取的追加写入到微扰的里面
write("./scf.xyz", extract, format='extxyz', append=True)
