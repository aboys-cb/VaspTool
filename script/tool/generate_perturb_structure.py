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
from hiphive.structure_generation import generate_mc_rattled_structures
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

    perturbed_system = system.perturb(pert_num=int(num * 0.4),
                                      cell_pert_fraction=0.05,
                                      atom_pert_distance=0.1,
                                      atom_pert_style='uniform')

    structures = perturbed_system.to('ase/structure')
    for structure in structures:
        structure.info['Config_type'] = "dpdata perturb 0.05 0.1"

    # append=True是追加写入 怕缓存影响  直接覆盖写入  如果有需要自己改成True
    write(f"./perturb_{system.formula}.xyz", structures, format='extxyz', append=True)

    rattle_std = 0.04
    min_distance = 0.1
    structures_mc_rattle = generate_mc_rattled_structures(
        system.to('ase/structure')[0], int(num * 0.6), rattle_std, min_distance, n_iter=20)
    for structure in structures_mc_rattle:
        structure.info['Config_type'] = "hiphive mc perturb 0.04 0.1"
    write(f"./perturb_{system.formula}.xyz", structures_mc_rattle, format='extxyz', append=True)
