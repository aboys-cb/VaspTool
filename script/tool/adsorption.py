#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/29 19:15
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
建模吸附模型
"""
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure, Molecule
from pymatgen.core.surface import SlabGenerator

# 这个是读取表面的原始结构 而不是slab模型
structure = Structure.from_file("../../Cu.cif")
# 可以读文件 也可以直接建立
# adsorption_molecule=Molecule.from_file("molecule.xyz")
adsorption_molecule = Molecule("HHO",
                               [[7.16750, 1.59835, 8.57334
                                 ],
                                [5.60698, 1.60212, 8.56915],
                                [6.38919, 2.17224, 8.40802]])
# adsorption_molecule.to("aaa.xyz", fmt="xyz")

slab = SlabGenerator(
    structure,
    miller_index=(1, 1, 1),  # 米勒指数
    min_slab_size=8,  # 最小slab
    min_vacuum_size=15  #真空层大小
).get_slab()
finder = AdsorbateSiteFinder(slab, selective_dynamics=True)
all = finder.generate_adsorption_structures(adsorption_molecule,
                                            (3, 3, 1),  # 扩包比例
                                            find_args={"distance": 2}  # 将吸附物放在表面上2A的位置
                                            )
for i, s in enumerate(all):
    s.to(f"{i}.vasp", fmt="poscar")
