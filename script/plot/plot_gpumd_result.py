#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 22:23
# @Author  : 兵
# @email    : 1747193328@qq.com
import os.path

import matplotlib

matplotlib.use('Agg')

from gpyumd.load import load_thermo
import matplotlib.pyplot as plt

if os.path.exists("thermo.out"):
    data = load_thermo()
    plt.plot(list(range(data["U"].shape[0])), data["U"])
    plt.savefig("./energy.png", dpi=150)
else:
    print("没有找到画图文件，请完善逻辑！")
