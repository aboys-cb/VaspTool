#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 16:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from pynep.calculate import NEP
from pynep.select import FarthestPointSample
from sklearn.decomposition import PCA

a = read('train.xyz', ':')
calc = NEP("nep.txt")
print(calc)
des = np.array([np.mean(calc.get_property('descriptor', i), axis=0) for i in a])
sampler = FarthestPointSample(min_distance=0.02)
selected_i = sampler.select(des, [])
print(len(selected_i))
write('selected.xyz', [a[i] for i in selected_i])

reducer = PCA(n_components=2)
reducer.fit(des)
proj = reducer.transform(des)
plt.scatter(proj[:, 0], proj[:, 1], label='all data')
selected_proj = reducer.transform(np.array([des[i] for i in selected_i]))
plt.scatter(selected_proj[:, 0], selected_proj[:, 1], label='selected data')
plt.legend()
plt.axis('off')
plt.savefig('select.png')
