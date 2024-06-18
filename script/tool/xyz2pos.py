#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/14 13:56
# @Author  : 兵
# @email    : 1747193328@qq.com
import sys

from ase.io import read, write

pos_path = sys.argv[1]
if len(sys.argv) == 3:
    index = sys.argv[2]
else:
    index = -1

write("POSCAR", read(pos_path, index=index, format="extxyz"))
