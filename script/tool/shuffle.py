#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 20:31
# @Author  : 兵
# @email    : 1747193328@qq.com
import sys

from ase.io import read, write
from sklearn.utils import shuffle

path = sys.argv[1]
atoms = read(path, ":", format="extxyz")

atoms = shuffle(atoms)
print("打乱成功！")
write(path, atoms, format='extxyz')
