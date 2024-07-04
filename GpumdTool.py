#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/7/2 11:17
# @Author  : 兵
# @email    : 1747193328@qq.com
import matplotlib

matplotlib.use("Agg")
import argparse
import datetime
import glob
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from ase.io import read as ase_read
from ase.io import write as ase_write
import matplotlib.pyplot as plt
from monty.os import cd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # 指定输出流为sys.stdout

)
# 这里是间距  每隔NumSamples 抽取一个
# 因为我是每1000步输出一个 跑1ns 一共也就100个数据  所以我设置2 抽取50
NumSamples = 2


def run(run_cmd: str, run_dir: Path):
    start = datetime.datetime.now()
    logging.info("\t开始计算")

    vasp_cmd = [os.path.expanduser(os.path.expandvars(run_cmd))]
    with cd(run_dir), open(f"{run_cmd}.out", "w") as f_std, open(f"{run_cmd}.err", "w", buffering=1) as f_err:
        subprocess.check_call(vasp_cmd, stdout=f_std, stderr=f_err)
    logging.info("\t计算完成" + f"\t耗时：{datetime.datetime.now() - start}")


def load_thermo(thermo_path):
    """
    因为gpyumd安装不方便  只用到了这个函数  就先复制过来
    """

    data = pd.read_csv(thermo_path, sep='\s+', header=None)
    labels = ['temperature', 'K', 'U']
    # Format before v3.3.1
    if data.shape[1] == 9:  # orthogonal
        labels += ['Px', 'Py', 'Pz', 'Lx', 'Ly', 'Lz']
    elif data.shape[1] == 15:  # triclinic
        labels += ['Px', 'Py', 'Pz', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']
    # format after v3.3.1
    elif data.shape[1] == 12:  # orthogonal
        labels += ['Px', 'Py', 'Pz', 'Pyz', 'Pxz', 'Pxy', 'Lx', 'Ly', 'Lz']
    elif data.shape[1] == 18:  # triclinic
        labels += ['Px', 'Py', 'Pz', 'Pyz', 'Pxz', 'Pxy', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']
    else:
        raise ValueError(f"The file {thermo_path} is not a valid thermo.out file.")

    out = dict()
    for i in range(data.shape[1]):
        out[labels[i]] = data[i].to_numpy(dtype='float')

    return out


def verify_path(path: Path) -> None:
    """
    会检查是否存在路径，若不存在，则创建该路径，支持多级目录创建
    :param path:
    :return:
    """
    if not path.exists():
        # path.mkdir()
        os.makedirs(path)


def cp_file(source_file: Path, destination_dir: Path) -> None:
    """
    复制文件
    :param source_file: 要复制的文件
    :param destination_dir: 希望复制到的路径
    :return:
    """
    src_files = glob.glob(source_file.as_posix())
    for i in src_files:
        logging.debug(f"\t复制文件：{i} -> {destination_dir.as_posix()}")
        shutil.copy(i, destination_dir.as_posix())
    return


def iter_path(glob_strs: list):
    def decorator(func):
        def wrapper(path, *args, **kwargs):
            for glob_str in glob_strs:

                for i in path.glob(glob_str):
                    try:
                        func(i, *args, **kwargs)
                    except KeyboardInterrupt:
                        return
                    except Exception:
                        pass

        return wrapper

    return decorator


@iter_path(["*.xyz", "*.vasp"])
def molecular_dynamics(path: Path):
    """
    根据指定的文件夹 以此计算文件夹下的所有的xyz文件

    :param self:
    :return:
    """

    if path.suffix == ".vasp":
        atoms = ase_read(path, 0, format="vasp")

    else:
        atoms = ase_read(path, 0, format="extxyz")
    md_path = root_path.joinpath(f"gpumd_cache/{atoms.symbols}/md")
    verify_path(md_path)
    logging.info(f"路径：{md_path.as_posix()}")

    cp_file(root_path.joinpath("run.in"), md_path.joinpath("run.in"))
    cp_file(root_path.joinpath("nep.txt"), md_path.joinpath("nep.txt"))
    atoms.write(md_path.joinpath("model.xyz"), format="extxyz")
    run("gpumd", md_path)

    data = load_thermo(md_path.joinpath("thermo.out"))
    fig = plt.figure()
    plt.plot(list(range(data["U"].shape[0])), data["U"])
    plt.savefig(md_path.joinpath("md_energy.png"), dpi=150)

    # 后处理 抽取指定数量的到result下

    dump_atoms = ase_read(md_path.joinpath("dump.xyz"), ":", format="extxyz")
    extract_atoms = dump_atoms[::NumSamples]
    logging.info(f"抽取了{len(extract_atoms)}个数据，保存在：./result/{atoms.symbols}.xyz")

    ase_write(root_path.joinpath(f"result/{atoms.symbols}.xyz"), extract_atoms, format="extxyz")


def prediction(self):
    pass


def build_argparse():
    parser = argparse.ArgumentParser(description="""GPUMD 工具. 
        可以批量md和主动学习 """,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "job_type", choices=["prediction", "md"], help=" "
    )
    parser.add_argument(
        "path", type=Path, help="要计算的xyz路径，或者要批量计算的文件夹。"
    )

    return parser


if __name__ == '__main__':
    # 采样
    parser = build_argparse()
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.mkdir("./result")
    root_path = Path("./")

    if args.job_type == "md":
        molecular_dynamics(args.path)
    elif args.job_type == "prediction":
        prediction(args.path)
