#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 12:09
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
绘制分子动力学的
"""
import sys

import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun

plt.style.use("./science.mplstyle")
# vasp_path=sys.argv[1]
plt.figure(figsize=(3.5, 2.625))
# vasp_path = "./vasprun.xml"
vasp_path = sys.argv[1]
vasprun = Vasprun(vasp_path, parse_potcar_file=False)

name = vasprun.final_structure.composition.to_pretty_string()

energies = [step["e_0_energy"] for step in vasprun.ionic_steps]
steps = list(range(1, len(energies) + 1))
plt.plot(steps, energies, label=name)
plt.ylabel("E0 Energy(eV)")
plt.xlabel("time(fs)")

plt.legend()
plt.tight_layout()
plt.savefig(f"./aimd-{name}.png", dpi=300)
