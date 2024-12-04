#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/27 17:16
# @Author  : 兵
# @email    : 1747193328@qq.com
import numpy as np
from ase.io import read as ase_read
from calorine.nep import get_descriptors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

config = [
    # (文件名,图例,图例颜色)
    # ("./dpdata.xyz","dpdata","red"),
    ("./train.xyz", "train", "blue")

]
#
# 画原子分布
distribution = "atom"
# 画结构分布
# distribution="structure"

fit_data = []

for info in config:

    atoms_list = ase_read(info[0], ":", format="extxyz", do_not_split_by_at_sign=True)
    # 原子分布
    if distribution == "atom":

        atoms_list_des = np.vstack([get_descriptors(i, "nep.txt") for i in atoms_list])
    # 结构分布
    else:

        atoms_list_des = np.vstack([np.mean(get_descriptors(i, "nep.txt"), axis=0) for i in atoms_list])

    fit_data.append(atoms_list_des)

reducer = PCA(n_components=2)
reducer.fit(np.vstack(fit_data))
fig = plt.figure()
for index, array in enumerate(fit_data):
    proj = reducer.transform(array)
    plt.scatter(proj[:, 0], proj[:, 1], label=config[index][1], c=config[index][2])

plt.legend()
plt.axis('off')

plt.savefig("./distribution.png")
