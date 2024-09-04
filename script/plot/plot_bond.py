#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/8/10 22:51
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
画原子键长变化的 临时写的
"""

# path=sys.argv[1]
import matplotlib.pyplot as plt
from ase.io import read as ase_read

path = "dump.xyz"
frames = ase_read(path, ":", format="extxyz")
bonds = []
for atoms in frames:
    # print(atoms[16])
    dis = atoms.get_distance(27, 55)

    bonds.append(dis)
plt.plot(list(range(len(bonds))), bonds)
plt.show()
