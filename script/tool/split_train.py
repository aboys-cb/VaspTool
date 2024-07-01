#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/14 12:00
# @Author  : 兵
# @email    : 1747193328@qq.com

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if Path("train-0.9.xyz").exists():
    print("当前目录下已经有train-0.9.xyz文件，将追加到文件，而不是覆盖写入。")
if Path("test-0.1.xyz").exists():
    print("当前目录下已经有train-0.9.xyz文件，将追加到文件，而不是覆盖写入。")
path = Path(sys.argv[1])
if path.is_file():
    files = [path]
else:
    files = []
    for file in path.glob("*.xyz"):
        files.append(file)
count = 0
for file in tqdm(files, "文件分割"):
    atoms_list = read(file, ":", format="extxyz")
    screen_list = []
    for atoms in atoms_list:

        if (np.any(abs(atoms.calc.results["forces"]) > 100)):
            continue
        screen_list.append(atoms)
    count += len(screen_list)
    train, test = train_test_split(screen_list, test_size=0.1, random_state=88, shuffle=True)
    # 这里append=True 考虑可以将多个体系合并下

    write("./train-0.9.xyz", train, format='extxyz', append=True)
    write("./test-0.1.xyz", test, format='extxyz', append=True)
print(f"数据集一共有{count}条")
