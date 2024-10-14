#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 22:39
# @Author  : 兵
# @email    : 1747193328@qq.com
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.core import Structure


def save_summary(summary):
    with open("ACF.dat", "w", encoding="utf8") as f:
        header = "Id,X,Y,Z,label,charge,transfer,min dist,atomic volume".split(",")

        header = [i.center(10) for i in header]
        header_text = "".join(header)
        f.write(header_text)
        f.write("\n")
        f.write("-" * 100)
        f.write("\n")
        structure = Structure.from_file("POSCAR")
        for index in range(len(structure)):
            site = structure[index]
            line = [index + 1, round(site.x, 4), round(site.y, 4), round(site.z, 4), site.label,
                    round(summary['charge'][index], 4),
                    round(summary['charge_transfer'][index], 4),
                    round(summary['min_dist'][index], 4),
                    round(summary['atomic_volume'][index], 4)]
            line = [str(i).center(10) for i in line]

            f.write("".join(line))
            f.write("\n")
        f.write("-" * 100)
        f.write("\n")

        f.write(f"vacuum charge :   {summary['vacuum_charge']}\n")
        f.write(f"vacuum volume :   {summary['vacuum_volume']}\n")
        f.write(f"bader version :   {summary['bader_version']}\n")


if __name__ == '__main__':
    print("开始bader电荷分析。")

    summary = bader_analysis_from_path("./")
    print("bader电荷分析完成。")

    save_summary(summary)
