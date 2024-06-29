#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 19:18
# @Author  : 兵
# @email    : 1747193328@qq.com

from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp.outputs import Vasprun

# 这一步是读取 XDATCAR，得到一系列结构信息
traj = Vasprun("./vasprun.xml").get_trajectory()
traj: Trajectory
# 这一步是实例化 DiffusionAnalyzer 的类
# 并用 from_structures 方法初始化这个类； 900 是温度，2 是POTIM 的值，1是间隔步数
# 间隔步数（step_skip）不太容易理解，但是根据官方教程:
# dt = timesteps * self.time_step * self.step_skip

diff = DiffusionAnalyzer.from_structures(traj, 'Ag', 300, 1, 10)

# 可以用内置的 plot_msd 方法画出 MSD 图像
# 有些终端不能显示图像，这时候可以调用 export_msdt() 方法，得到数据后再自己作图
# diff.plot_msd()
# plt.show()
