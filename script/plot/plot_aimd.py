#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 12:09
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
绘制分子动力学的
"""

import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun

plt.style.use("./science.mplstyle")
# vasp_path=sys.argv[1]
vasp_path = "./vasprun.xml"
vasprun = Vasprun(vasp_path, parse_potcar_file=False)

name = vasprun.final_structure.composition.to_pretty_string()
energies = [step["total"] for step in vasprun.ionic_steps]
steps = list(range(1, len(energies) + 1))
plt.plot(steps, energies, label=name)

plt.legend()
plt.tight_layout()
plt.savefig(f"./aimd-{name}.png", dpi=300)
