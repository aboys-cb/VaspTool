#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 14:32
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
等间距抽取多少个结构在xyz文件中
"""
from ase.io import read, write

file_path = "./train.xyz"
atoms = read(file_path, index=":", format="extxyz")

extract = atoms[::40]
print(len(extract))
# 这里将抽取的追加写入到微扰的里面
write("./scf.xyz", extract, format='extxyz', append=True)
