#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 11:07
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
脚本执行方式：python generate_perturb_structure.py some_structure_path num
some_structure_path 可以是POSCAR、CONTCAR、.vasp、.xyz文件
num是生成微扰结构的个数
"""
import sys

import dpdata
from ase.io import write

path = sys.argv[1]
num = int(sys.argv[2])
perturbed_system = dpdata.System(path).perturb(pert_num=num,
                                               cell_pert_fraction=0.1,
                                               atom_pert_distance=0.5,
                                               atom_pert_style='normal')

structures = perturbed_system.to('ase/structure')
# append=True是追加写入 怕缓存影响  直接覆盖写入  如果有需要自己改成True
write("./scf.xyz", structures, format='extxyz', append=False)
