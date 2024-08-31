#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/8/28 15:14
# @Author  : 兵
# @email    : 1747193328@qq.com
# 按照间隔分割xyz 分散任务 多节点提交

# python split_xyz.py new.xyz 10
import sys

from ase.io import read, write

job_num = int(sys.argv[2])
atoms_list = read(sys.argv[1], index=":", format="extxyz", do_not_split_by_at_sign=True)


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


result = split_list(atoms_list, job_num)

for i, sublist in enumerate(result):
    write(f"split-{i}-num-{len(sublist)}.xyz", sublist)
