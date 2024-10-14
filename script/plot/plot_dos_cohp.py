#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/5/18 19:19
# @Author  : 兵
# @email    : 1747193328@qq.com
from itertools import product
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import palettable
from matplotlib.patches import ConnectionPatch
from numpy._typing import ArrayLike
from pymatgen.electronic_structure.cohp import Cohp, CompleteCohp
from pymatgen.electronic_structure.core import Spin, Orbital, OrbitalType
from pymatgen.electronic_structure.dos import Dos
from pymatgen.io.vasp import Vasprun

plt.style.use("./science.mplstyle")
class DosCohpPlotter:

    def __init__(self, zero_at_efermi=True):
        self.figure = plt.figure( )
        self.stack=False
        self.zero_at_efermi = zero_at_efermi
        self._doses: dict[
            str, dict[Literal["energies", "densities", "efermi"], float | ArrayLike | dict[Spin, ArrayLike]]
        ] = {}
        self._cohps: dict[str, dict[str, np.ndarray | dict[Spin, np.ndarray] | float]] = {}

    def add_dos(self, label, dos:Dos):
        """Add a dos for plotting.
        从其他地方粘贴的

        Args:
            label: label for the DOS. Must be unique.
            dos: Dos object
        """
        if dos.norm_vol is None:
            self._norm_val = False
        energies = dos.energies - dos.efermi if self.zero_at_efermi else dos.energies
        densities = dos.densities
        efermi = dos.efermi
        self._doses[label] = {
            "energies": energies,
            "densities": densities,
            "efermi": efermi,
        }

    def add_cohp(self, label, cohp:Cohp):
        """Add a COHP for plotting.
        从其他地方粘贴的
        Args:
            label: Label for the COHP. Must be unique.

            cohp: COHP object.
        """
        energies = cohp.energies - cohp.efermi if self.zero_at_efermi else cohp.energies
        populations = cohp.get_cohp()
        int_populations = cohp.get_icohp()
        self._cohps[label] = {
            "energies": energies,
            "COHP": populations,
            "ICOHP": int_populations,
            "efermi": cohp.efermi,
        }
    @staticmethod
    def get_orb_list(orb: str):
        """

        :param orb: str 4d or 5p
        :return:
        """
        result = []
        for i in Orbital:
            if str(i.orbital_type) == orb[1:]:
                result.append(orb[:1] + i.name)
        return result

    def compose_orbit(self,orb):
        """
        对传入的轨道进行拆分组合
        :param orb: 4d-5p or 4d-5px or 4dx2-5p
        :return:
        """
        a, b = orb.split("-")
        a_orb = [a] if a[-1] not in ["s", "p", "d", "f"] else self.get_orb_list(a)
        b_orb = [b] if b[-1] not in ["s", "p", "d", "f"] else self.get_orb_list(b)

        result = []
        for a, b in product(a_orb, b_orb):
            result.append(f"{a}-{b}")
        return result



    def parse_config(self,dos_config:dict, cohp_config:dict):
        """
        解析下投影配置文件 将需要画图的放在字典里
        :param dos_config: dict
        :param cohp_config: dict
        :return:
        Examples
        -----
        dos_conf = {"vasprun_path": "../cache/Cs1Ag0.5Bi0.5I3/vasprun.xml",
         "projected": {"I": ["p"],"Ag": [ "d"],"Bi": ["p" ]   },
         }

        cohp_conf={
            "cohpcar_path":"../cache/Cs1Ag0.5Bi0.5I3/COHPCAR.lobster",
            "poscar_path":"../cache/Cs1Ag0.5Bi0.5I3/POSCAR",
            "projected": {"I": ["p"], "Ag": ["d"], "Bi": ["p"]}

        }
        plotter=DosCohpPlotter()
        plotter.parse_config(dos_conf,cohp_conf)

        """
        #解析dos的
        orb_map = ["s", "p", "d", "f"]
        vasprun = Vasprun(dos_config["vasprun_path"], parse_potcar_file=False)
        #加入总的dos 先不加入  主要看投影
        # self.add_dos("total", vasprun.tdos)

        for elem, orbits in dos_config["projected"].items():

            if isinstance(elem, int):
                site = vasprun.final_structure[elem - 1]
                elem = site.label

                elem_dos = vasprun.complete_dos.get_site_spd_dos(site)

            else:
                elem_dos = vasprun.complete_dos.get_element_spd_dos(elem)

            for orb in orbits:
                orb_type = OrbitalType(orb_map.index(orb))

                self.add_dos(f"{elem}-{orb}", elem_dos[orb_type])
        #解析cohp

        complete_cohp = CompleteCohp.from_file(filename=cohp_config["cohpcar_path"], fmt='LOBSTER',
                                           structure_file=cohp_config["poscar_path"])

        for elem_label, config in cohp_config["projected"].items():
            if isinstance(config["label"], tuple):

                label = [str(i) for i in range(config["label"][0], config["label"][1] + 1)]
            else:
                label = config["label"]
            cohp=None
            for orb in config["orb"]:

                for _orb in self.compose_orbit(orb):
                    # complete_cohp.get_summed_cohp_by_label_list()
                    _cohp = complete_cohp.get_summed_cohp_by_label_and_orbital_list(label,[_orb] * len(label))
                    if cohp is None:
                        cohp=_cohp
                    else:
                        #对轨道进行加和

                        if Spin.up  in cohp.cohp.keys():

                            cohp.cohp[Spin.up]+=_cohp.cohp[Spin.up]
                        if Spin.down in cohp.cohp.keys():
                            cohp.cohp[Spin.down] += _cohp.cohp[Spin.down]


            if cohp:
                self.add_cohp(elem_label, cohp)


    def get_plot(self, energy_lim=(-2, 2), density_lim=(-10, 10), cohp_lim=(-5,5), invert_axes=False):

        if invert_axes:
            #反转 竖排模式 左边为Dos 右边为Cohp
            pass
            gridspec = self.figure.add_gridspec(1, 2,
                                                wspace=0.1 ,
                                                width_ratios=[1,1],

                                                )

        else:
            #上下堆叠 上面为Dos  下面为Cohp
            gridspec = self.figure.add_gridspec(2, 1,
                                                 hspace=0.1 ,
                                                height_ratios=[1,1],
                                                )


        #先画Dos
        dos_axes=self.figure.add_subplot(gridspec[0])

        n_colors = min(9, max(3, len(self._doses)))

        colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

        all_pts = []
        idx=0
        for   idx, key in enumerate(self._doses.keys()):
            for spin in [Spin.up, Spin.down]:
                if spin in self._doses[key]["densities"]:
                    energy = self._doses[key]["energies"]
                    densities = list(int(spin) * self._doses[key]["densities"][spin])
                    if invert_axes:
                        x = densities
                        y = energy
                    else:
                        x = energy
                        y = densities
                    all_pts.extend(list(zip(x, y)))
                    if self.stack:
                        dos_axes.fill(x, y, color=colors[idx % n_colors], label=str(key))

                    else:
                        dos_axes.plot(x, y, color=colors[idx % n_colors], label=str(key) )
        # 画cohp
        cohp_axes = self.figure.add_subplot(gridspec[1])
        n_colors = min(9, max(3, len(self._cohps)))

        for idx, key in enumerate(self._cohps.keys()):
            energies = self._cohps[key]["energies"]
            populations = self._cohps[key]["COHP"]
            for spin in [Spin.up, Spin.down]:
                if spin in populations:
                    if invert_axes:
                        x = -populations[spin]
                        y = energies
                    else:
                        x = energies
                        y = -populations[spin]

                    if spin == Spin.up:
                        cohp_axes.plot(
                            x,
                            y,
                            color=colors[idx % n_colors],
                            linestyle="-",
                            label=str(key),

                        )
                    else:
                        cohp_axes.plot(x, y, color=colors[idx % n_colors], linestyle="--", linewidth=3)



        cohp_axes.tick_params(axis='both', which='both', direction='in')
        dos_axes.tick_params(axis='both', which='both', direction='in')
        energy_label = "$E - E_f$ (eV)" if self.zero_at_efermi else "Energy (eV)"
        energy_label="Energy (eV)"
        if invert_axes:
            #画一个水平线

            con = ConnectionPatch(xyA=(density_lim[0],0), xyB=(cohp_lim[1],0), coordsA="data", coordsB="data",
                                  axesA=dos_axes, axesB=cohp_axes, color="k",linestyle="--", linewidth=0.5)

            cohp_axes.add_artist(con)
            cohp_axes.text(0.1 , 0.1, 'Antibonding', transform=cohp_axes.transAxes,rotation="vertical" ,   color='k')
            cohp_axes.text(0.8, 0.16, 'Bonding', transform=cohp_axes.transAxes,rotation="vertical" ,   color='k')

            # cohp_axes.set_xticklabels([])

            cohp_axes.set_yticklabels([])
            cohp_axes.set_xlim(cohp_lim)
            cohp_axes.set_ylim(energy_lim)
            cohp_axes.axvline(x=0, color="k", linestyle="-", linewidth=0.5)



            handles, labels = cohp_axes.get_legend_handles_labels()
            label_dict = dict(zip(labels, handles))
            cohp_axes.legend(label_dict.values(), label_dict, loc="upper right" )

            cohp_axes.set_xlabel("-COHP")


            # dos_axes.set_xticklabels([])
            dos_axes.axvline(x=0, color="k", linestyle="-", linewidth=0.5 )

            dos_axes.set_xlim(density_lim)
            dos_axes.set_ylim(energy_lim)
            dos_axes.set_ylabel(energy_label)

            dos_axes.set_xlabel("DOS (states/eV)")
            handles, labels = dos_axes.get_legend_handles_labels()
            label_dict = dict(zip(labels, handles))
            dos_axes.legend(label_dict.values(), label_dict, loc="upper right" )

        else:

            con = ConnectionPatch(xyA=( 0,density_lim[1]), xyB=(0,cohp_lim[0]), coordsA="data", coordsB="data",
                                  axesA=dos_axes, axesB=cohp_axes, color="k",linestyle="--")

            cohp_axes.add_artist(con)
            cohp_axes.text(0.2 , 0.1, 'Antibonding', transform=cohp_axes.transAxes,  color='k')
            cohp_axes.text(0.2 , 0.7, 'Bonding', transform=cohp_axes.transAxes,   color='k')


            # cohp_axes.set_yticklabels([])


            cohp_axes.axhline(y=0, color="k", linestyle="-" )

            cohp_axes.set_ylim(cohp_lim)
            cohp_axes.set_xlim(energy_lim)
            cohp_axes.set_ylabel("-COHP")
            cohp_axes.set_xlabel(energy_label)

            dos_axes.set_xticklabels([])
            # dos_axes.set_yticklabels([])

            dos_axes.set_xlim(energy_lim)
            dos_axes.set_ylim(density_lim)
            dos_axes.axhline(y=0, color="k", linestyle="-" )

            dos_axes.set_ylabel("DOS (states/eV)")
            handles, labels = dos_axes.get_legend_handles_labels()
            label_dict = dict(zip(labels, handles))
            dos_axes.legend(label_dict.values(), label_dict,ncols=2,   loc="upper right" )
            handles, labels = cohp_axes.get_legend_handles_labels()
            label_dict = dict(zip(labels, handles))
            cohp_axes.legend(label_dict.values(), label_dict,ncols=2, loc="upper right"  )



        #如果边框太多空白 调整这里
        plt.subplots_adjust(left=0.1, right=0.9 ,bottom=0.1, top=0.9  )


if __name__ == '__main__':
    # dos_conf = {"vasprun_path": "../cache/Cs1Ag0.5Bi0.5I3/vasprun.xml",
    #      "projected": {"I": ["p"],"Ag": [ "d"],"Bi": ["s","p" ]   },
    #      }
    #
    # cohp_conf={
    #     "cohpcar_path":"../cache/Cs1Ag0.5Bi0.5I3/COHPCAR.lobster",
    #     "poscar_path":"../cache/Cs1Ag0.5Bi0.5I3/POSCAR",
    #     "projected": {"Bi(6s)-I(5p)":{
    #                             "label":(185,190),
    #                             "orb":["6s-5p"]
    #                         },
    #                     "Bi(6p)-I(5p)": {
    #                         "label": (185, 190),
    #                         "orb": ["6p-5p"]
    #                     },
    #                     "Ag(4d)-I(5p)": {
    #                         "label": (161, 166),
    #                         "orb": ["4d-5p"]
    #                     }
    #     }
    #
    # }

    sb_dos_conf = {"vasprun_path": "../cache/Cs8Ag4Bi3Sb1I24/vasprun.xml",
         "projected": {"I": ["p"],"Ag": [ "d"],"Bi": ["s","p" ] , "Sb": ["s","p" ] },
         }

    sb_cohp_conf={
        "cohpcar_path":"../cache/Cs8Ag4Bi3Sb1I24/COHPCAR.lobster",
        "poscar_path":"../cache/Cs8Ag4Bi3Sb1I24/POSCAR",
        "projected": {"Bi(6s)-I(5p)":{
                                "label":(185,190),
                                "orb":["6s-5p"]
                            },
                        "Bi(6p)-I(5p)": {
                            "label": (185, 190),
                            "orb": ["6p-5p"]
                        },

            "Sb(5s)-I(5p)": {
                "label": (203, 208),
                "orb": ["5s-5p"]
            },
            "Sb(5p)-I(5p)": {
                "label": (203, 208),
                "orb": ["5p-5p"]
            },
                        "Ag(4d)-I(5p)": {
                            "label": (161, 166),
                            "orb": ["4d-5p"]
                        }
        }

    }
    # cu_dos_conf = {"vasprun_path": "../cache/Cu/vasprun.xml",
    #             "projected": {"I": ["p"], "Ag": ["d"], "Bi": ["s", "p"], "Cu": ["d"]},
    #             }
    #
    # cu_cohp_conf = {
    #     "cohpcar_path": "../cache/Cu/COHPCAR.lobster",
    #     "poscar_path": "../cache/Cu/POSCAR",
    #     "projected": {"Bi(6s)-I(5p)": {
    #         "label": (185, 190),
    #         "orb": ["6s-5p"]
    #     },
    #         "Bi(6p)-I(5p)": {
    #             "label": (185, 190),
    #             "orb": ["6p-5p"]
    #         },
    #
    #         "Cu(4d)-I(5p)": {
    #             "label": (161, 166),
    #             "orb": ["3d-5p"]
    #         },
    #         "Ag(4d)-I(5p)": {
    #             "label": (167, 172),
    #             "orb": ["4d-5p"]
    #         }
    #     }
    #
    # }

    # 这里可以是分轨道  比如"6px-5px" 如果不是分轨道  会把所有的加和
    plotter=DosCohpPlotter()
    plotter.parse_config(sb_dos_conf,sb_cohp_conf)
    plotter.get_plot(invert_axes=True,cohp_lim=(-10,20),energy_lim=(-2,2),density_lim=(0,10))
    plt.savefig("dos_and_cohp_sb.png")