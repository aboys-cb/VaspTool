#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 22:37
# @Author  : 兵
# @email    : 1747193328@qq.com
from gpyumd.load import load_thermo
from pylab import *

matplotlib.use('Agg')

data = load_thermo()
plot(list(range(data["U"].shape[0])), data["U"])
savefig("./en.png", dpi=150)
