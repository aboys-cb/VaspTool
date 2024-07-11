#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 16:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from calorine.nep import get_descriptors
from pynep.calculate import NEP
from pynep.select import FarthestPointSample
from sklearn.decomposition import PCA

atoms_list = read('dump.xyz', ':')
print(len(atoms_list))
screen_list = []
for atoms in atoms_list:
    # if (np.any(abs(atoms.calc.results["forces"]) > 50)):
    #     continue
    screen_list.append(atoms)
print(len(screen_list))
calc = NEP("nep.txt")
des = np.array([np.mean(get_descriptors(i, "nep.txt"), axis=0) for i in screen_list])

# des = np.array([np.mean(calc.get_property('descriptor', i), axis=0) for i in screen_list])
sampler = FarthestPointSample(min_distance=0.01)
selected_i = sampler.select(des, min_select=0)
print(len(selected_i))
# for i in tqdm.tqdm(selected_i):
#     write('selected.xyz', screen_list[i], append=True)

reducer = PCA(n_components=2)
reducer.fit(des)
proj = reducer.transform(des)
plt.scatter(proj[:, 0], proj[:, 1], label='all data')
selected_proj = reducer.transform(np.array([des[i] for i in selected_i]))
plt.scatter(selected_proj[:, 0], selected_proj[:, 1], label='selected data')
plt.legend()
plt.axis('off')
plt.savefig('select.png')
