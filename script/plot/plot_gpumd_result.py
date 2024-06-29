#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 22:23
# @Author  : 兵
# @email    : 1747193328@qq.com
from gpyumd.load import load_kappa
from gpyumd.math import running_ave
from pylab import *

matplotlib.use('Agg')

kappa = load_kappa()
t = np.arange(1, kappa['kxi'].shape[0] + 1) * 0.001  # ns
kappa['kyi_ra'] = running_ave(kappa['kyi'], t)
kappa['kyo_ra'] = running_ave(kappa['kyo'], t)
kappa['kxi_ra'] = running_ave(kappa['kxi'], t)
kappa['kxo_ra'] = running_ave(kappa['kxo'], t)
kappa['kz_ra'] = running_ave(kappa['kz'], t)
figure(figsize=(12, 10))
subplot(2, 2, 1)
# set_fig_properties([gca()])
plot(t, kappa['kyi'], color='C7', alpha=0.5)
plot(t, kappa['kyi_ra'], linewidth=2)
xlim([0, 1.25])
# gca().set_xticks(range(0,11,2))
ylim([-2000, 4000])
gca().set_yticks(range(-2000, 4001, 1000))
xlabel('time (ns)')
ylabel(r'$\kappa_{in}$ W/m/K')
title('(a)')

subplot(2, 2, 2)
# set_fig_properties([gca()])
plot(t, kappa['kyo'], color='C7', alpha=0.5)
plot(t, kappa['kyo_ra'], linewidth=2, color='C3')
xlim([0, 1.25])
# gca().set_xticks(range(0,11,2))
ylim([0, 4000])
gca().set_yticks(range(0, 4001, 1000))
xlabel('time (ns)')
ylabel(r'$\kappa_{out}$ (W/m/K)')
title('(b)')

subplot(2, 2, 3)

plot(t, kappa['kyi_ra'], linewidth=2)
plot(t, kappa['kyo_ra'], linewidth=2, color='C3')
plot(t, kappa['kyi_ra'] + kappa['kyo_ra'], linewidth=2, color='k')
xlim([0, 1.25])
# gca().set_xticks(range(0,11,2))
ylim([0, 4000])
gca().set_yticks(range(0, 4001, 1000))
xlabel('time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend(['in', 'out', 'total'])
title('(c)')
subplot(2, 2, 4)

plot(t, kappa['kyi_ra'] + kappa['kyo_ra'], color='k', linewidth=2)
plot(t, kappa['kxi_ra'] + kappa['kxo_ra'], color='C0', linewidth=2)
plot(t, kappa['kz_ra'], color='C3', linewidth=2)
xlim([0, 1.25])
# gca().set_xticks(range(0,11,2))
ylim([0, 4000])
gca().set_yticks(range(-2000, 4001, 1000))
xlabel('time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend(['yy', 'xy', 'zy'])
title('(d)')

tight_layout()
savefig("./result.png", dpi=150)
