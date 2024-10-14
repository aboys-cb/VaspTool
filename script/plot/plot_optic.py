#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 13:07
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
绘制光吸收曲线的图
"""
import matplotlib.pyplot as plt
from pymatgen.analysis.solar.slme import absorption_coefficient, optics, slme
from pymatgen.io.vasp.outputs import  Vasprun

plt.style.use("./science.mplstyle")
fig=plt.figure()
sort_name=[
    ("$Cs_2AgBiI_6$", "./Cs1Ag0.5Bi0.5I3.xml", 0.85),
    ("$Cs_2Cu_{0.25}Ag_{0.75}BiI_6$", "./Cs1Cu0.125Ag0.375Bi0.5I3.xml", 0.4618),
    ("$Cs_2AgBi_{0.75}Sb_{0.25}I_6$", "./Cs2AgBi0.75Sb0.25I6.xml", 0.5952)

]

for label, path, gap in sort_name:
    vasp=Vasprun(path)
    new_en, new_abs =absorption_coefficient(vasp.dielectric)

    new_en += gap
    plt.plot(new_en, new_abs,label=label)
    data = optics(path)

    print(data[2], data[3], slme(*data, thickness=5e-6))

plt.legend(ncol=2)
# plt.ylim(0,7)
# plt.ticklabel_format(style='sci', scilimits=(0,0))
plt.xlim(0, 5)
plt.xlabel("Photon energy (eV)")

plt.ylabel("Absorption ($cm^{-1}$)")
plt.yscale('log')
plt.savefig("./absorption_coefficient.png")