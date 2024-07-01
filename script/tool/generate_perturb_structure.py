#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 11:07
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
脚本执行方式：python generate_perturb_structure.py some_structure_path num
some_structure_path 可以是POSCAR、CONTCAR、.vasp、 文件
num是生成微扰结构的个数
"""
import sys
from pathlib import Path

import dpdata
from ase.io import write
from tqdm import tqdm

path = Path(sys.argv[1])
if path.is_file():
    files = [path]
else:
    files = []
    for file in path.glob("POSCAR"):
        files.append(file)

    for file in path.glob("*/POSCAR"):
        files.append(file)

num = int(sys.argv[2])
for file in tqdm(files):
    system = dpdata.System(file, "vasp/poscar")
    perturbed_system = system.perturb(pert_num=num,
                                      cell_pert_fraction=0.05,
                                      atom_pert_distance=0.1,
                                                                  atom_pert_style='uniform')

    structures = perturbed_system.to('ase/structure')
    for structure in structures:
        structure.info['Config_type'] = "perturb 0.05 0.1"

    # append=True是追加写入 怕缓存影响  直接覆盖写入  如果有需要自己改成True
    write(f"./perturb_{system.formula}.xyz", structures, format='extxyz', append=True)
