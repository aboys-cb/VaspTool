#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 19:23
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

from pymatgen.core import Structure

path = "./"
if not os.path.exists("./super"):
    os.mkdir("./super")
for cif in os.listdir(path):
    if os.path.isfile(cif) and cif.endswith("cif"):
        struct = Structure.from_file(cif)
        supercell = struct.make_supercell([2, 1, 1])
        supercell.to("./super/" + supercell.composition.to_pretty_string() + ".cif")
