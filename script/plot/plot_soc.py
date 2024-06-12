#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 22:40
# @Author  : 兵
# @email    : 1747193328@qq.com
import itertools
import re
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from monty.io import zopen

plt.rc('font', family='Times New Roman')
# 修改公式中默认字体
from matplotlib import rcParams

rcParams['mathtext.default'] = 'regular'
import matplotlib as mpl
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import BSVasprun



class Procar:

    def __init__(self, filename):
        """
        Args:
            filename: Name of file containing PROCAR.
        """
        headers = None

        with zopen(filename, "rt") as file_handle:
            preambleexpr = re.compile(r"# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of " r"ions:\s*(\d+)")
            kpointexpr = re.compile(r"^k-point\s+(\d+).*weight = ([0-9\.]+)")
            bandexpr = re.compile(r"^band\s+(\d+)")
            ionexpr = re.compile(r"^ion.*")
            expr = re.compile(r"^([0-9]+)\s+")
            current_kpoint = 0
            current_band = 0
            done = False
            spin = Spin.down
            weights = None
            # pylint: disable=E1137
            for line in file_handle:
                # print(line)
                line = line.strip()
                if bandexpr.match(line):
                    m = bandexpr.match(line)
                    # print(m.group())
                    current_band = int(m.group(1)) - 1
                    current_direction = -1
                    done = False
                elif kpointexpr.match(line):
                    m = kpointexpr.match(line)
                    # print(m.groups())
                    current_kpoint = int(m.group(1)) - 1
                    weights[current_kpoint] = float(m.group(2))
                    if current_kpoint == 0:
                        spin = Spin.up if spin == Spin.down else Spin.down
                    done = False
                elif headers is None and ionexpr.match(line):
                    headers = line.split()
                    headers.pop(0)
                    # headers.pop(-1)

                    data = defaultdict(lambda: np.zeros((nkpoints, nbands, nions, len(headers))))

                    phase_factors = defaultdict(
                        lambda: np.full(
                            (nkpoints, nbands, nions, 3, len(headers)),
                            np.NaN,
                            dtype=np.float32,
                        )
                    )
                elif expr.match(line):
                    # print(line)
                    toks = line.split()
                    index = int(toks.pop(0)) - 1
                    # toks.pop(-1)
                    num_data = np.array([float(t) for t in toks[: len(headers)]])
                    # print(done)
                    if not done:
                        data[spin][current_kpoint, current_band, index, :] = num_data
                    else:

                        # for orb in range(len(["x","y","z"])):
                        phase_factors[spin][current_kpoint, current_band, index, current_direction, :] = num_data

                elif line.startswith("tot"):
                    # print("tot")
                    current_direction += 1
                    done = True
                elif preambleexpr.match(line):
                    m = preambleexpr.match(line)
                    nkpoints = int(m.group(1))
                    nbands = int(m.group(2))
                    nions = int(m.group(3))
                    weights = np.zeros(nkpoints)

            self.nkpoints = nkpoints
            self.nbands = nbands
            self.nions = nions
            self.weights = weights
            self.orbitals = headers
            self.data = data
            self.phase_factors = phase_factors

    def get_projection_on_elements(self, structure):
        """
        Method returning a dictionary of projections on elements.

        Args:
            structure (Structure): Input structure.

        Returns:
            a dictionary in the {Spin.up:[k index][b index][{Element:values}]]
        """
        dico = {}
        for spin in self.data:
            dico[spin] = [[defaultdict(float) for i in range(self.nkpoints)] for j in range(self.nbands)]

        for iat in range(self.nions):
            name = structure.species[iat].symbol
            for spin, d in self.data.items():
                # print(d.shape)
                for k, b in itertools.product(range(self.nkpoints), range(self.nbands)):
                    dico[spin][b][k][name] = np.sum(d[k, b, iat, :])
                # return

        return dico

    def get_spin_component_by_direction(self, direction="z"):
        directions = ["x", "y", "z"]
        if direction not in directions:
            print("只支持x y z三个方向")
            return
        direction_index = directions.index(direction)
        dico = {}
        for spin in self.data:
            dico[spin] = [[defaultdict(float) for i in range(self.nkpoints)] for j in range(self.nbands)]
            for k, b in itertools.product(range(self.nkpoints), range(self.nbands)):
                dico[spin][b][k] = np.sum(self.phase_factors[spin][k, b, :, direction_index, :], 0)[-1]
        # print(self.phase_factors[spin][k, b, :, direction_index, :])
        # print( (np.sum(self.phase_factors[spin][k, b, :, direction_index, :],0) ))
        return dico

    def get_occupation(self, atom_index, orbital):
        """
        Returns the occupation for a particular orbital of a particular atom.

        Args:
            atom_num (int): Index of atom in the PROCAR. It should be noted
                that VASP uses 1-based indexing for atoms, but this is
                converted to 0-based indexing in this parser to be
                consistent with representation of structures in pymatgen.
            orbital (str): An orbital. If it is a single character, e.g., s,
                p, d or f, the sum of all s-type, p-type, d-type or f-type
                orbitals occupations are returned respectively. If it is a
                specific orbital, e.g., px, dxy, etc., only the occupation
                of that orbital is returned.

        Returns:
            Sum occupation of orbital of atom.
        """
        orbital_index = self.orbitals.index(orbital)
        return {
            spin: np.sum(d[:, :, atom_index, orbital_index] * self.weights[:, None]) for spin, d in self.data.items()
        }

def get_ticks(bs):
    """
    Get all ticks and labels for a band structure plot.

    Returns:
        dict: A dictionary with 'distance': a list of distance at which
        ticks should be set and 'label': a list of label for each of those
        ticks.
    """

    ticks, distance = [], []
    for br in bs.branches:

        start, end = br["start_index"], br["end_index"]
        # print(br["name"])

        labels = br["name"].split("-")
        labels=[i for i in labels if i.strip()]
        # skip those branches with only one point
        if labels[0] == labels[1]:
            continue

        # add latex $$
        for idx, label in enumerate(labels):
            if label.startswith("\\") or "_" in label:
                labels[idx] = "$" + label + "$"
        if ticks and labels[0] != ticks[-1]:
            ticks[-1] += "$\\mid$" + labels[0]
            ticks.append(labels[1])
            distance.append(bs.distance[end])
        else:
            ticks.extend(labels)
            distance.extend([bs.distance[start], bs.distance[end]])

    return {"distance": distance, "label": ticks}
def plot_spin_by_direction(path_dir,direction,
        energy_min: float = -1,
        energy_max: float = 1,):
    bs_vasprun = BSVasprun(path_dir+"/vasprun.xml", parse_projected_eigen=True)
    pro = Procar(path_dir+"/PROCAR")
    projection_on_elements = pro.get_spin_component_by_direction(direction)
    band_structure = bs_vasprun.get_band_structure(line_mode=True)
    ware1,enery1,spin1 = [],[],[]
    ware2,enery2,spin2 = [],[],[]
    for band, projection in zip(band_structure.bands[Spin.up], projection_on_elements[Spin.up]):
        for distance, energy, tot in zip(band_structure.distance, band, projection):

            if tot >0:
                ware1.append(distance)
                enery1.append(energy - band_structure.efermi)
                spin1.append(tot)
            else:
                ware2.append(distance)
                enery2.append(energy - band_structure.efermi)
                spin2.append(tot)
    fig = plt.figure(figsize=(8,5))
    norm = mpl.colors.Normalize(-1,1)

    plt.plot([0, max(band_structure.distance)], [0, 0], 'k-.', linewidth=1)


    xticks =  get_ticks(band_structure)

    for dis in xticks["distance"]:

        plt.plot([dis,dis],[energy_min, energy_max],'k-.', linewidth=1)

    plt.xticks(xticks["distance"],xticks["label"])
    plt.xlim(0, max(xticks["distance"]))

    a = plt.scatter(ware1, enery1,c=spin1,s=30,lw=0, alpha=0.5 ,cmap=mpl.cm.coolwarm,norm=norm, marker="o")
    b = plt.scatter(ware2, enery2,c=spin2,s=20,lw=0, alpha=0.5,cmap=mpl.cm.coolwarm, norm=norm,marker="*")
    plt.ylim(energy_min, energy_max)
    plt.tick_params(axis='y', direction='in')
    plt.colorbar( fraction=0.2, pad=0.1)

    # plt.legend((a,b),("spin-up","spin-down"),fontsize=16 , frameon=False )
    plt.tight_layout()
    ax = plt.gca()
    #处理刻度
    ax.tick_params(labelsize=16,bottom=False, top=False, left=True, right=False)
    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.15, wspace=0.01, hspace=0.1)
    plt.xlabel("Wavevector $k$", fontsize=16 )
    plt.ylabel("$E-E_F$ / eV", fontsize=16 )
    # plt.title("title",x=0.5,y=1.02)
    # plt.savefig("bnd.eps",format='eps', transparent=True,bbox_inches='tight', dpi=600)
    plt.savefig("band.jpg",bbox_inches='tight', dpi=1200)


if __name__ == '__main__':
    #这里传入的是vasprun.xml所在的路径
    plot_spin_by_direction("./danzi/vasprun/",
                           "z",
                           -2,2)