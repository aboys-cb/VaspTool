#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/24 19:44
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
使用方式 python plot_sr_energy_force.py OUTCAR
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp.outputs import Outcar

path = sys.argv[1]
print("正在载入文件。。。")

out = Outcar(path)
print("开始解析能量。。。")
out.read_pattern({
    "e_fr_energy": r"free  energy   TOTEN\s+=\s+([\d\-\.]+)",
}, postprocess=float)

energy = np.array(out.data["e_fr_energy"])

energy = energy.flatten()
print("开始解析力。。。")

a = out.read_table_pattern(r"TOTAL-FORCE \(eV/Angst\)\n\s*\-+\n", r"\s+".join([r"(\-*[\.\d]+)"] * 6), r"-*\n",
                           last_one_only=False, postprocess=float)

force = np.array(a)[:, :, 3:]
force = force.reshape((force.shape[0], -1))
max_froce = np.max(force, 1)

result = np.vstack([np.arange(energy.shape[0]), energy, max_froce]).T
print("正在画图。。。")

fig, axes = plt.subplots(2, 1, sharex=True)
axes1, axes2 = axes
axes1.plot(result[:, 0], result[:, 1], label="energy", color="red")
axes1.set_ylabel("energy(eV)")
axes1.legend()

axes2.plot(result[:, 0], result[:, 2], label="max force", color="green")
axes2.set_ylabel("max force")
axes2.legend()

axes2.set_xlabel("steps")
plt.tight_layout()
plt.savefig("energy_forces.png", dpi=150)
np.savetxt("energy_forces.csv", result, header="step,energy,force", fmt='%.8f', comments="")
print("导出成功！./energy_forces.csv")
