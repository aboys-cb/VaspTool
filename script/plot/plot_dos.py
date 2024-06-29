#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 22:40
# @Author  : 兵
# @email    : 1747193328@qq.com
import matplotlib
import numpy as np

matplotlib.use("Agg")
import palettable
from matplotlib import pyplot as plt
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Vasprun

plt.style.use("./science.mplstyle")



class MyDosPlotter(DosPlotter):

    def get_plot(
            self,
            xlim=None,
            ylim=None,
            ax=None,
            invert_axes=False,
            beta_dashed=False,

    ):
        n_colors = min(9, max(3, len(self._doses)))

        colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

        ys = None
        all_densities = []
        all_energies = []
        for dos in self._doses.values():
            energies = dos["energies"]
            densities = dos["densities"]
            if not ys:
                ys = {
                    Spin.up: np.zeros(energies.shape),
                    Spin.down: np.zeros(energies.shape),
                }
            new_dens = {}
            for spin in [Spin.up, Spin.down]:
                if spin in densities:
                    if self.stack:
                        ys[spin] += densities[spin]
                        new_dens[spin] = ys[spin].copy()
                    else:
                        new_dens[spin] = densities[spin]
            all_energies.append(energies)
            all_densities.append(new_dens)

        keys = list((self._doses))
        # all_densities.reverse()
        # all_energies.reverse()
        all_pts = []

        for idx, key in enumerate(keys):
            for spin in [Spin.up, Spin.down]:
                if spin in all_densities[idx]:
                    energy = all_energies[idx]
                    densities = list(int(spin) * all_densities[idx][spin])
                    if invert_axes:
                        x = densities
                        y = energy
                    else:
                        x = energy
                        y = densities
                    all_pts.extend(list(zip(x, y)))
                    if self.stack:
                        ax.fill(x, y, color=colors[idx % n_colors], label=str(key))
                    elif spin == Spin.down and beta_dashed:
                        ax.plot(x, y, color=colors[idx % n_colors], label=str(key), linestyle="--" )
                    else:
                        ax.plot(x, y, color=colors[idx % n_colors], label=str(key) )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        elif not invert_axes:
            xlim = ax.get_xlim()
            relevant_y = [p[1] for p in all_pts if xlim[0] < p[0] < xlim[1]]
            ax.set_ylim((min(relevant_y), max(relevant_y)))
        if not xlim and invert_axes:
            ylim = ax.get_ylim()
            relevant_y = [p[0] for p in all_pts if ylim[0] < p[1] < ylim[1]]
            ax.set_xlim((min(relevant_y), max(relevant_y)))

        if self.zero_at_efermi:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(xlim, [0, 0], "k--" ) if invert_axes else ax.plot([0, 0], ylim, "k--" )

        if invert_axes:
            ax.axvline(x=0, color="k", linestyle="-" )
            # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, integer=True))
            # ax.yaxis.set_major_locator(ticker.MaxNLocator( integer=True))
            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        else:

            # ax.xaxis.set_major_locator(ticker.MaxNLocator( integer=True))
            # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2, integer=True))
            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

            ax.axhline(y=0, color="k", linestyle="-" )
        # ax.tick_params(axis='both', which='both', direction='in')
        # ax.tick_params(axis='both', which='both', direction='in')
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.tick_params(labelsize=16)
        # Remove duplicate labels with a dictionary
        handles, labels = ax.get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        ax.legend(label_dict.values(), label_dict, frameon=False,  ncol=2,  columnspacing=1 )


    def plot_all(self, dos_conf, invert_axes=True, energy_lim=None, density_lim=None):
        orb_map = ["s", "p", "d", "f"]





        if invert_axes:
            xlim, ylim = density_lim, energy_lim
            fig, axes = plt.subplots(1, len(dos_conf),   sharex=True, sharey=True)


        else:
            xlim, ylim = energy_lim, density_lim

            fig, axes = plt.subplots(len(dos_conf), 1,   sharex=True, sharey=True)
        if len(dos_conf)==1:
            axes=[axes]



        axes:list[plt.Axes]

        for col, conf in enumerate(dos_conf):
            vasprun = Vasprun(conf["path"], parse_potcar_file=False)

            # self.add_dos("total", vasprun.tdos)

            for elem, orbits in conf["projected"].items():

                if isinstance(elem,int):
                    site=vasprun.final_structure[elem-1]
                    elem=site.label

                    elem_dos = vasprun.complete_dos.get_site_spd_dos(site)

                else:
                    elem_dos = vasprun.complete_dos.get_element_spd_dos(elem)


                for orb in orbits:
                    orb_type = OrbitalType(orb_map.index(orb))

                    self.add_dos(f"{elem}-{orb}", elem_dos[orb_type])
            self.get_plot(xlim, ylim, ax=axes[col], invert_axes=invert_axes)

            if invert_axes:
                if col == 0:
                    axes[0].set_ylabel("Energy (eV)")

                axes[col].set_xlabel("DOS (states/eV)" )
            else:
                if col == len(dos_conf) - 1:
                    axes[col].set_xlabel("Energy (eV)")

                axes[col].set_ylabel("DOS (states/eV)" )

            self._doses.clear()

        plt.tight_layout(h_pad=0)


if __name__ == '__main__':
    plotter = MyDosPlotter()
    dos_conf = [

        {"path": "./vasprun.xml",
         "projected": {"I": ["p"], "Ag": ["d"], "Bi": ["p"]},
         },

    ]
    plotter.plot_all(dos_conf, energy_lim=(-2, 2), density_lim=(-10, 10), invert_axes=False)
    plt.savefig("./dos.png", dpi=300)
