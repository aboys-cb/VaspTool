#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 20:18
# @Author  : 兵
# @email    : 1747193328@qq.com
import sys

from ase.io import read, write

pos_path = sys.argv[1]
write("model.xyz", read(pos_path), format="extxyz")
