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

structure = Structure.from_file("../../Cu.cif")
# adsorption_molecule=Molecule.
adsorption_molecule = Molecule("HHO",
                               [[7.16750, 1.59835, 8.57334
                                 ],
                                [5.60698, 1.60212, 8.56915],
                                [6.38919, 2.17224, 8.40802]])
adsorption_molecule.to("aaa.xyz", fmt="xyz")

slab = SlabGenerator(
    structure,
    miller_index=(1, 1, 1),
    min_slab_size=8,
    min_vacuum_size=15
).get_slab()
finder = AdsorbateSiteFinder(slab, selective_dynamics=True)
all = finder.generate_adsorption_structures(adsorption_molecule,
                                            (3, 3, 1),
                                            find_args={"distance": 2.5})
for i, s in enumerate(all):
    s.to(f"{i}.vasp", fmt="poscar")
