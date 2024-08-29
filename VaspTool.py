#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 22:40
# @Author  : 兵
# @email    : 1747193328@qq.com
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # 指定输出流为sys.stdout

)
__version__ = "1.2.0"
logging.info(f"VaspTool-{__version__}")

logging.info(f"开始初始化，请稍等...")


from functools import cached_property, partial


import matplotlib

matplotlib.use('Agg')

from ruamel.yaml.comments import CommentedMap
import abc
import argparse
import glob
import re
import shutil
import warnings
from pathlib import Path
import numpy as np
import json
import traceback
import pandas as pd
import datetime
import os
import subprocess
from typing import *
from tqdm import tqdm

from monty.os import cd
from monty.dev import requires
from monty.io import zopen
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, Lattice, SETTINGS

from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, VaspInput, Potcar, PotcarSingle
from pymatgen.io.vasp.outputs import Vasprun, BSVasprun, Outcar, Eigenval, Wavecar, Locpot

from pymatgen.io.lobster import Lobsterin, Lobsterout, Icohplist
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.dos import CompleteDos

from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter, BSDOSPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.analysis.solar import slme
from pymatgen.analysis.eos import EOS

from pymatgen.io.ase import AseAtomsAdaptor


try:
    from phonopy import Phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS, write_disp_yaml, write_FORCE_SETS
    from phonopy.interface.calculator import get_default_physical_units
    from phonopy.interface.phonopy_yaml import PhonopyYaml
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
    from pymatgen.io import phonopy
except:
    Phonopy = None


try:
    from ase.io import read as ase_read
    from ase.io import write as ase_write
except:
    ase_write = None
    ase_read = None




from matplotlib import pyplot as plt



if os.path.exists("./config.yaml"):
    conf_path = "./config.yaml"
elif Path(__file__).with_name("config.yaml").exists():
    conf_path = Path(__file__).with_name("config.yaml").as_posix()
else:
    logging.error("在运行路径或者VaspTool.py路径下必须要有一个config.yaml!")
    exit()
logging.info(f"使用配置文件：{conf_path}")

config = loadfn(conf_path)
config: CommentedMap
SETTINGS["PMG_DEFAULT_FUNCTIONAL"] = r"PBE_54"

SETTINGS["PMG_VASP_PSP_DIR"] = os.path.expanduser(os.path.expandvars(config["SETTING"]["PMG_VASP_PSP_DIR"]))

plt.rc('font', family='Times New Roman')

warnings.filterwarnings("ignore", module="pymatgen")


PotcarSingle.functional_dir["PBE_54"] = ""
FUNCTION_TYPE = ["pbe", "pbesol", "hse", "scan", "r2scan", "mbj", "gw", "bse"]
KPOINTS_TYPE = Union[int, tuple, list]
setting = config.get("SETTING", {})

potcar_config = config.get("POTCAR", {}).get("PBE54")

potcar_gw_config = config.get("POTCAR", {}).get("GW")

# step_base_incar 是基于pbe最基本的设置 其他泛函需要更改的在function_base_incar
step_base_incar = {
    "sr": {
        "add": {
            "LWAVE": False, "LCHARG": False, "NSW": 100, "ISIF": 3, "IBRION": 2, "ALGO": "Normal"
        },
        "remove": []
    },
    "scf": {
        "add": {
            "LWAVE": True, "LCHARG": True, "NSW": 0, "IBRION": -1
        },
        "remove": []
    },


    "dos": {
        "add": {
            "ISTART": 1, "ISMEAR": 0, "ICHARG": 11, "NSW": 0, "IBRION": -1, "LORBIT": 11,
            "NEDOS": 3000, "LWAVE": False, "LCHARG": False
        },
        "remove": []
    },
    "band": {
        "add": {
            "ISTART": 1, "ICHARG": 11, "NSW": 0, "IBRION": -1, "LORBIT": 11, "LWAVE": False, "LCHARG": False
        },
        "remove": []
    },

    "optic": {
        "add": {
            "ISTART": 1, "NSW": 0, "LWAVE": False,
            "LCHARG": False, "LOPTICS": True, "NBANDS": 96,
            "NEDOS": 2000, "CSHIF": 0.100, "IBRION": 8
        },
        "remove": []
    },
    "elastic": {
        "add": {
            "ISTART": 0, "ISIF": 3, "IBRION": 6, "LWAVE": False, "LCHARG": False,
            "PREC": "Accurate", "ADDGRID": True, "LREAL": False, "NSW": 1,
            "NFREE": 2
        },
        "remove": ["NPAR", "NCORE"]
    },

    "dielectric": {
        "add": {
            "ISTART": 1, "SIGMA": 0.05, "LEPSILON": True, "LPEAD": True, "IBRION": 8, "LWAVE": False, "LCHARG": False
        },
        "remove": ["NPAR", "NCORE"]
    },
    "aimd": {
        "add": {
            "ALGO": "Normal", "IBRION": 0, "MDALGO": 2, "ISYM": 0,
            "POTIM": 1, "NSW": 3000, "TEBEG": 300, "TEEND": 300,
            "SMASS": 1, "LREAL": "Auto", "ISIF": 2, "ADDGRID": True
        },
        "remove": []
    },

}
# 这个都是非pbe的一些补充
function_base_incar = {

    "hse": {
        "base": {
            "add": {
                    "HFSCREEN": 0.2, "AEXX": 0.25, "LHFCALC": True, "PRECFOCK": "N"
                    },
            "remove": []
        },

        "steps": {
            "scf": {
                "ISTART": 1, "ALGO": "Damped", "ICHARG": 0
            },

            "dos": {"ALGO": "Normal", "ICHARG": 1,
                    },
            "band": {
                "ALGO": "Normal",
                "ICHARG": 1,
            },
            "optic": {"ICHARG": 2, "LREAL": False, "ALGO": "Normal", "IBRION": -1}

        }
    },
    "pbesol": {
        "base": {
            "add": {"GGA": "PS"},
            "remove": []
        },
        "steps": {

        }
    },
    "scan": {
        "base": {
            "add": {"METAGGA": "SCAN", "ALGO": "ALL", "LASPH": True,
                    "LUSE_VDW": True, "BPARAM": 15.7, "CPARAM": 0.0093},
            "remove": ["GGA"]
        },
        "steps": {
            "scf": {"ALGO": "ALL", "ICHARG": 2},
            "dos": {"ICHARG": 1},
            "band": {"ICHARG": 1},

        }
    },
    "r2scan": {
        "base": {
            "add": {"METAGGA": "R2SCAN", "LASPH": True,
                    "LUSE_VDW": True, "BPARAM": 11.95, "CPARAM": 0.0093},
            "remove": ["GGA"]
        },
        "steps": {

            "scf": {"ALGO": "ALL", "ICHARG": 2},
            "dos": {"ICHARG": 1},
            "band": {"ICHARG": 1, "LREAL": False, },

        }
    },

    "mbj": {
        "base": {
            "add": {"ALGO": "Exact", "LOPTICS": True,
                    "CSHIFT": 0.1, "NEDOS": 2000, "ISTART": 1},
            "remove": ["GGA"]
        },
        "steps": {

            "dos": {"ICHARG": 2},
            "band": {"ICHARG": 1},

        }

    },
    "gw": {
        "base": {
            "add": {"ALGO": "EVGW0", "LSPECTRAL": True, "NELMGW": 1,
                    "ISTART": 1, "LOPTICS": True, "LREAL": False
                    },
            "remove": ["NPAR", "NCORE"]
        },
        "steps": {

        }

    },
    "bse": {
        "base": {
            "add": {"ALGO": "BSE", "LSPECTRAL": True, "NELMGW": 1,
                    "ISTART": 1, "LOPTICS": True, "LREAL": False,
                    "NBANDSO": 4, "NBANDSV": 20, "OMEGAMAX": 60
                    },
            "remove": ["NPAR", "NCORE"]
        },
        "steps": {

        }

    },
}

def hash_file(obj, file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = f.read()
    hash1 = hash(data)
    hash2 = hash(str(obj))
    return hash1 == hash2


def get_pot_symbols(species, mode: Literal["pbe54", "gw"] = "pbe54"):
    """
    根据传入 返回赝势列表
    :param species:
    :param mode:
    :return:
    """
    symbols = []
    for i in species:

        if mode == "pbe54":

            v = potcar_config[i.name]
        elif mode == "gw":

            v = potcar_gw_config[i.name]
        else:
            break
        if symbols:

            if symbols[-1] == v:
                continue
        symbols.append(v)

    return symbols


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


def get_command_path(command_name):
    return get_command_result(['which', command_name])


def get_command_result(cmd):
    try:
        # 使用 subprocess 调用 which 命令，并捕获输出
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 检查命令是否成功执行

        if result.returncode == 0:
            # 返回命令的路径
            return result.stdout.strip()
        else:
            # 如果命令未找到，返回 None 或抛出异常
            return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


def check_in_out_file(path):
    in_out_file = ["INCAR", "POSCAR", "KPOINTS", "POTCAR", "OUTCAR"]
    return all([os.path.exists(os.path.join(path, i)) for i in in_out_file])


def array_to_dat(file_path, x_array, *data_array, headers: list = []):
    x_array = x_array.reshape(1, -1)
    all_data_array = np.array(data_array)
    result = None

    with open(file_path, "w", encoding="utf8") as f:

        f.write("# " + "".join(map(lambda x: f"{x:<15}", headers)) + '\n')

        f.write(f"# GroupSize & Groups: {all_data_array.shape[2]}  {all_data_array.shape[1]}"  '\n')

        for i in range(all_data_array.shape[1]):
            if i % 2 == 0:

                single = np.vstack([x_array, all_data_array[:, i, :]])
            else:
                single = np.vstack([np.flip(x_array), np.flip(all_data_array[:, i, :], axis=1)])

            f.write('\n')

            for row in single.T:
                f.write("".join(map(lambda x: f"{x:<15.8f} " if not isinstance(x, dict) else f"{x} ", row)) + '\n')
        # np.savetxt(f,single.T, delimiter="    ", fmt='%f',comments="",header=header)
# 将xyz 获取的
def write_to_xyz(vaspxml_path, save_path, Config_type, append=True):
    if setting.get("ExportXYZ"):
        if ase_read is None:
            logging.error("设置开启了导出xyz文件，但没有安装ase，请 pip install ase")
        else:
            atoms_list = []
            atoms = ase_read(vaspxml_path, index=":")
            index = 1
            for atom in atoms:
                xx, yy, zz, yz, xz, xy = -atom.calc.results['stress'] * atom.get_volume()  # *160.21766
                atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])

                atom.calc.results['energy'] = atom.calc.results['free_energy']

                atom.info['Config_type'] = Config_type + str(index)
                atom.info['Weight'] = 1.0
                del atom.calc.results['stress']
                del atom.calc.results['free_energy']
                atoms_list.append(atom)
                index += 1

            ase_write(save_path, atoms_list, format='extxyz', append=append)


def store_dataframe_as_json(dataframe, filename, orient="split"):
    with zopen(filename, "w") as f:
        data = json.dumps(dataframe.to_dict(orient=orient), cls=MontyEncoder)
        f.write(data)


def load_dataframe_from_json(filename, pbar=True, decode=True):
    # Progress bar for reading file with hook
    pbar1 = tqdm(desc=f"Reading file {filename}", position=0, leave=True, ascii=True, disable=not pbar)
    # Progress bar for decoding objects
    pbar2 = tqdm(desc=f"Decoding objects from {filename}", position=0, leave=True, ascii=True, disable=not pbar)

    with zopen(filename, "rb") as f:
        dataframe_data = json.load(f, cls=MontyDecoder)

    pbar1.close()
    pbar2.close()

    if isinstance(dataframe_data, dict):
        if set(dataframe_data.keys()) == {"data", "columns", "index"}:
            return pd.DataFrame(**dataframe_data)
    else:
        return pd.DataFrame(dataframe_data)


def read_dataframe_from_file(file_path: Path, duplicated=True, **kwargs) -> pd.DataFrame:
    """
    从指定路径读取结构 可以是文件夹路径、结构路径

    Returns: (pd.DataFrame)
    """
    if file_path.is_dir():

        systems = []
        for p in file_path.iterdir():

            try:
                s = read_dataframe_from_file(p, False)
                systems.append(s)
            except:
                logging.warning(f"读取结构文件{p}失败。")
                pass
        df = pd.concat(systems)

    else:

        if file_path.suffix.endswith(".json"):
            df = load_dataframe_from_json(file_path, **kwargs)
        elif file_path.name.endswith("POSCAR") or file_path.suffix in [".cif", ".vasp"]:
            struct = Structure.from_file(file_path)
            struct.remove_oxidation_states()
            if setting.get("UseInputFileName", False):
                system = file_path.stem
            else:
                system = struct.composition.to_pretty_string()
            df = pd.DataFrame([{"system": system,
                                "structure": struct}])
        elif file_path.name.endswith("xyz"):
            systems = []
            if ase_read is None:
                logging.error("xyz文件必须安装ase,请 pip install ase 安装！")
                return pd.DataFrame()
            atoms = ase_read(file_path, index=":", format="extxyz")
            for atom in atoms:
                struct = AseAtomsAdaptor.get_structure(atom)
                # xyz 分子式一样 所以加个数字标识下
                systems.append({"system": struct.composition.to_pretty_string(),
                                "structure": struct})
            df = pd.DataFrame(systems)

        else:
            raise ValueError(f"仅支持后缀为POSCAR、cif、vasp、json、xyz类型的文件")

    if duplicated:
        df.reset_index(drop=True, inplace=True)


        duplicated = df[df.duplicated("system", False)]

        group = duplicated.groupby("system")

        df["group_number"] = group.cumcount()
        df["group_number"] = df["group_number"].fillna(-1)
        df["group_number"] = df["group_number"].astype(int)
        df['system'] = df.apply(
            lambda row: f"{row['system']}-{row['group_number'] + 1}" if row['group_number'] >= 0 else row['system'],
            axis=1)
        df.drop("group_number", inplace=True, axis=1)
    df.reset_index(drop=True, inplace=True)
    return df


def verify_path(path: Path) -> None:
    """
    会检查是否存在路径，若不存在，则创建该路径，支持多级目录创建
    :param path:
    :return:
    """
    if not path.exists():
        # path.mkdir()
        os.makedirs(path)


def get_vacuum_axis(structure: Structure, vacuum_size=10):
    """
    判断真空层所在的轴  大于5A的控件被判定为真空轴
    没有返回None
    :param structure:
    :return:
    """
    coords = np.array([site.coords for site in structure.sites])
    maxcoords = np.max(coords, axis=0)
    mincoords = np.min(coords, axis=0)

    if (structure.lattice.a - maxcoords[0]) + (mincoords[0]) > vacuum_size:
        return 0
    elif (structure.lattice.b - maxcoords[1]) + (mincoords[1]) > vacuum_size:
        return 1
    elif (structure.lattice.c - maxcoords[2]) + (mincoords[2]) > vacuum_size:
        return 2
    else:
        return None


class BaseIncar(Incar):
    PBE_EDIFF = 1e-06
    PBE_EDIFFG = -0.01
    HSE_EDIFF = 1e-04
    HSE_EDIFFG = -0.01
    ENCUT = 500

    def __init__(self, params: dict = None, **kwargs):
        super().__init__(params)

        self.update(kwargs)

    @classmethod
    def build(cls, system: str, function: FUNCTION_TYPE = "pbe", **kwargs):

        base = config.get("INCAR").copy()
        base: dict
        # 不同泛函的基本参数
        # 关于杂化泛函 ICHARG=1比ICHARG=2快一倍  但是能量稍微差一点
        # Si2 hse dos
        # ICHARG=1: CBM:6.3352 VBM:5.3661  dos_gap:0.9691 耗费时间：30min
        # ICHARG=2: CBM:6.3218 VBM:5.3525  dos_gap:0.9693 耗费时间：12min

        # ---------------------------------------------------------------------
        step_incar = step_base_incar[system]

        base.update(step_incar["add"])
        for i in step_incar["remove"]:
            if i in base:
                base.pop(i)

        if function != "pbe":
            function_incar = function_base_incar[function]
            base.update(function_incar["base"]["add"])
            for i in function_incar["base"]["remove"]:
                if i in base:
                    base.pop(i)
            step_function_incar = function_incar["steps"].get(system)
            if step_function_incar:
                base.update(step_function_incar)
        base.update(kwargs)

        return cls(base)

    def has_magnetic(self, structure):
        """
        根据元素周期表判断体系是否具有磁性，如果有就打开自旋。

        :return: 返回(bool,str)
        """

        magmom = []
        spin = []
        _ = [0, 0]
        for site in structure.sites:
            if site.species_string in config.get("MAGMOM").keys():
                mag = config.get("MAGMOM")[site.species_string]
                spin.append(True)
            elif site.specie.name in config.get("MAGMOM").keys():
                mag = config.get("MAGMOM")[site.specie.name]
                spin.append(True)
            else:
                mag = 0
                spin.append(False)
            if _[1] == mag:
                _[0] += 1
            else:
                magmom.append(f"{_[0]}*{_[1]}")
                _ = [1, mag]
        magmom.append(f"{_[0]}*{_[1]}")
        if any(spin):
            self["ISPIN"] = 2
            self["MAGMOM"] = " ".join(magmom)

    def auto_encut(self, structure: Structure, pseudopotential="pbe54"):
        max_encut = 0

        for symbol in get_pot_symbols(structure.species, pseudopotential):
            single = PotcarSingle.from_symbol_and_functional(symbol, functional="PBE_54")

            if max_encut < single.enmax:
                max_encut = single.enmax
        encut = int(setting.get("ENCUTScale") * max_encut)
        self["ENCUT"] = encut
        logging.info(f"\t截断能根据{setting.get('ENCUTScale')}倍取值：{encut}")



class BaseKpoints:
    _instance = None
    init_flag = False

    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, kpoints_type="Gamma"):
        if BaseKpoints.init_flag:
            return

        BaseKpoints.init_flag = True
        self.kpoints_type = kpoints_type

        self.kpoints = config.get("KPOINTS")

    def get_kpoint_setting(self, job_type: str, step_type: str, function: str):

        if job_type not in self.kpoints.keys():
            return 30
        if step_type not in self.kpoints[job_type].keys():
            return 30
        if function not in self.kpoints[job_type][step_type].keys():
            function = "default"
        return self.kpoints[job_type][step_type][function]

    def get_kpoints(self, job_type: str, step_type: str, function: str, structure: Structure):
        kp = self.get_kpoint_setting(job_type, step_type, function)
        if isinstance(kp, int):
            if kp >= 100:
                kp = Kpoints.automatic_density(structure, kp).kpts[0]
            else:

                kps = [kp, kp, kp]
                vacuum = get_vacuum_axis(structure, 10)

                if vacuum is not None:
                    kps[vacuum] = 1

                kp = Kpoints.automatic_density_by_lengths(structure, kps).kpts[0]
        logging.info(f"\t网格K点：{kp}")

        if self.kpoints_type.upper().startswith("M"):
            return Kpoints.monkhorst_automatic(kp)
        return Kpoints.gamma_automatic(kp)

    def get_line_kpoints(self, path: Path, function: str, structure: Structure, job_type="band_structure",
                         step_type="band") -> Kpoints:
        if function == "pbe":
            if os.path.exists("./HIGHPATH"):
                kpoints = Kpoints.from_file("./HIGHPATH")
                logging.info("使用自定义的高对称路径文件！")
                # 下面这个循环 是将伽马点转换希腊字符，画图时的用
                for i, k in enumerate(kpoints.labels):

                    if "gamma" in k.lower():
                        kpoints.labels[i] = "$\\Gamma$"
            else:

                kpath = HighSymmKpath(structure, path_type="hinuma")
                kpoints = Kpoints.automatic_linemode(self.get_kpoint_setting(job_type, step_type, function), kpath)
                # 下面这个循环 是将伽马点转换希腊字符，画图时的用
                for i, k in enumerate(kpoints.labels):
                    if "gamma" in k.lower():
                        kpoints.labels[i] = "$\\Gamma$"

            return kpoints

        if path.joinpath("pbe/band/vasprun.xml").exists():

            pbe_vasprun = BSVasprun(path.joinpath("pbe/band/vasprun.xml").as_posix())
            pbe_kpoints = Kpoints.from_file(path.joinpath("pbe/band/KPOINTS").as_posix())
            kpoints1 = Kpoints.from_file(path.joinpath("pbe/scf/IBZKPT").as_posix())

            kpoints = Kpoints("Generated by VaspTool ", kpoints1.num_kpts + len(pbe_vasprun.actual_kpoints),
                              style=Kpoints.supported_modes.Reciprocal,
                              kpts=kpoints1.kpts + pbe_vasprun.actual_kpoints,
                              kpts_weights=kpoints1.kpts_weights + [0 for i in range(len(pbe_vasprun.actual_kpoints))])
            lables = []
            for k in kpoints.kpts:
                if k in pbe_kpoints.kpts:
                    lables.append(pbe_kpoints.labels[pbe_kpoints.kpts.index(k)])
                else:
                    lables.append(None)
            kpoints.labels = lables
            return kpoints
        else:
            kpts: list[float | None] = []
            weights: list[float | None] = []
            all_labels: list[str | None] = []
            kp = self.get_kpoint_setting(job_type, "scf", function)
            if isinstance(kp, int):
                grid = Kpoints.automatic_density(structure, kp).kpts[0]
            else:
                grid = kp
            ir_kpts = SpacegroupAnalyzer(structure, symprec=0.1).get_ir_reciprocal_mesh(grid)
            for k in ir_kpts:
                kpts.append(k[0])
                weights.append(int(k[1]))
                all_labels.append(None)

            # for line mode only, add the symmetry lines w/zero weight

            kpath = HighSymmKpath(structure, path_type="hinuma")
            frac_k_points, labels = kpath.get_kpoints(
                line_density=self.get_kpoint_setting(job_type, step_type, function), coords_are_cartesian=False
            )

            for k, f in enumerate(frac_k_points):
                kpts.append(f)
                weights.append(0.0)
                all_labels.append(labels[k])

            comment = "run along symmetry lines"

            return Kpoints(
                comment=comment,
                style=Kpoints.supported_modes.Reciprocal,
                num_kpts=len(kpts),
                kpts=kpts,  # type: ignore
                kpts_weights=weights,
                labels=all_labels,
            )


class JobBase():
    result_label = []

    def __init__(self, structure: Structure, path, job_type, step_type, function, kpoints_type="Gamma", folder=None,
                 KPOINTS=None,
                 open_soc=False, dft_u=False, force_coverage=False, mpirun_path="mpirun", vasp_path="vasp_std", cores=1,
                 **kwargs):
        self.test = None
        self.structure = structure
        self.path: Path = path
        self.job_type = job_type
        self.step_type = step_type
        if folder is None:
            self.folder = self.step_type
        else:
            self.folder = folder
        self.function = function
        self.open_soc = open_soc
        self.dft_u = dft_u
        self.kpoints_type = kpoints_type
        if KPOINTS is not None:
            assert isinstance(KPOINTS, Kpoints), f"自定义KPOINTS必须传入一个Kpoints对象而不是{type(KPOINTS)}"
        self.KPOINTS = KPOINTS
        self.force_coverage = force_coverage
        self.mpirun_path = mpirun_path
        self.vasp_path = vasp_path
        self.cores = cores
        self.cb_energy = 4
        self.dpi = 300
        self.vb_energy = -4
        self.incar_kwargs = {}
        for k, v in kwargs.items():
            if k.isupper():
                # 暂且把全大写的分配到incar 后面有bug再说
                self.incar_kwargs[k] = v
            else:
                setattr(self, k, v)

        # 要计算的类型 比如能带
        # 要计算的类型的细分步骤 优化 自洽 性质等
        verify_path(self.run_dir)
        logging.info("当前计算路径：" + self.run_dir.as_posix())
        if self.function in ["gw"]:
            self.pseudopotential = "gw"
        else:
            self.pseudopotential = "pbe54"

    @cached_property
    def run_dir(self) -> Path:
        """
        获取vasp 计算路径
        :return:
        """
        if self.test is not None:
            return self.path.joinpath(f"{self.function}/{self.folder}/{self.test}")

        return self.path.joinpath(f"{self.function}/{self.folder}")

    @cached_property
    def incar(self) -> BaseIncar:
        """Incar object."""

        incar = BaseIncar.build(self.step_type, self.function)
        formula = self.structure.composition.to_pretty_string()
        incar["SYSTEM"] = formula + "-" + self.function + "-" + self.step_type
        incar.has_magnetic(self.structure)

        if setting.get("ENCUTScale"):
            incar.auto_encut(self.structure)


        incar.update(self.incar_kwargs)
        if self.open_soc:
            incar["LSORBIT"] = True
        if self.dft_u and incar.get("LDAU") is None:
            data_u = config.get("U", {})

            if not data_u:
                logging.warning("\t开启DFT+U必须在配置文件设置U,开启失败!")
                return incar
            LDAUL = []
            LDAUU = []
            LDAUJ = []
            LDAUL_max = 1
            for elem in self.structure.composition.elements:
                if elem.name in data_u.keys():
                    LDAUL.append(str(data_u[elem.name]["LDAUL"]))
                    LDAUU.append(str(data_u[elem.name]["LDAUU"]))
                    LDAUJ.append(str(data_u[elem.name]["LDAUJ"]))
                    if LDAUL_max < data_u[elem.name]["LDAUL"]:
                        LDAUL_max = data_u[elem.name]["LDAUL"]
                else:

                    LDAUL.append("-1")
                    LDAUU.append("0")
                    LDAUJ.append("0")

            if all([i == "-1" for i in LDAUL]):
                logging.warning("\t在配置文件中没有找到该体系的U值,开启失败!")
                return incar
            incar["LDAU"] = True
            incar["LDAUTYPE"] = 2
            incar["LMAXMIX"] = LDAUL_max * 2
            incar["LDAUL"] = " ".join(LDAUL)
            incar["LDAUU"] = " ".join(LDAUU)
            incar["LDAUJ"] = " ".join(LDAUJ)

        return incar

    @cached_property
    def kpoints(self) -> Kpoints:
        """Kpoints object."""
        if self.KPOINTS is None:
            return BaseKpoints(self.kpoints_type).get_kpoints(self.job_type, self.step_type, self.function,
                                                              self.structure)
        else:
            return self.KPOINTS

    @cached_property
    def poscar(self) -> Poscar:
        """Poscar object."""
        poscar = Poscar(self.structure)
        return poscar

    @cached_property
    def potcar(self) -> Potcar:
        potcar = Potcar(symbols=get_pot_symbols(self.structure.species, self.pseudopotential), functional="PBE_54")
        return potcar

    def check_cover(self):
        """
        检查输入文件 避免重复计算 如果不需要重复计算 返回True 否则返回False
        :param run_dir:
        :return:
        """
        if not self.force_coverage and check_in_out_file(self.run_dir):
            hash_table = [
                hash_file(self.incar, self.run_dir.joinpath("INCAR")),
                hash_file(self.kpoints, self.run_dir.joinpath("KPOINTS")),
                hash_file(self.poscar, self.run_dir.joinpath("POSCAR")),
                hash_file(self.potcar, self.run_dir.joinpath("POTCAR")),
            ]
            if all(hash_table):
                try:
                    if Outcar(os.path.join(self.run_dir, "OUTCAR")).run_stats.get("User time (sec)"):
                        logging.info("\t已有缓存，如果覆盖运行，设置--force_coverage 或者 -f ")

                        return True
                except:
                    pass
        src_files = ["WAVE*", "CHG*", "*.tmp"]
        for src in src_files:

            src_file_list = self.run_dir.glob(src)
            for file in src_file_list:
                Path(file).unlink()

        return False

    def run(self, timeout=None, lobster=None, remove_wavecar=False):
        if self.open_soc:
            # 如果打开了soc 并且 scf  或band in
            vasp_path = self.vasp_path.with_name("vasp_ncl")
        else:
            vasp_path = self.vasp_path
        vasp_input = VaspInput(self.incar, self.kpoints, self.poscar, self.potcar)
        vasp_cmd = [self.mpirun_path, "-np", str(self.cores), vasp_path]

        start = datetime.datetime.now()
        logging.info("\t开始计算")
        vasp_input.write_input(output_dir=self.run_dir)
        if lobster:
            lobster.write_INCAR(incar_input=self.run_dir.joinpath("INCAR"), incar_output=self.run_dir.joinpath("INCAR"),
                                poscar_input=self.run_dir.joinpath("POSCAR"))
        vasp_cmd = vasp_cmd or SETTINGS.get("PMG_VASP_EXE")  # type: ignore[assignment]
        if not vasp_cmd:
            raise ValueError("No VASP executable specified!")
        vasp_cmd = [os.path.expanduser(os.path.expandvars(t)) for t in vasp_cmd]
        if not vasp_cmd:
            raise RuntimeError("You need to supply vasp_cmd or set the PMG_VASP_EXE in .pmgrc.yaml to run VASP.")
        with cd(self.run_dir), open("vasp.out", "w") as f_std, open("vasp.err", "w", buffering=1) as f_err:
            subprocess.check_call(vasp_cmd, stdout=f_std, stderr=f_err, timeout=timeout)
        logging.info("\t计算完成" + f"\t耗时：{datetime.datetime.now() - start}")
        if remove_wavecar:
            self.run_dir.joinpath("WAVECAR").unlink()

        return self

    @abc.abstractmethod
    def post_processing(self, result=None):
        pass



class StructureRelaxationJob(JobBase):
    """
    结构优化的类
    """

    def __init__(self, **kwargs):
        super().__init__(step_type="sr", **kwargs)
        # vasp 有时会让复制contcar 继续优化  这个是控制复制次数
        self.run_count = 3

    def run(self, **kwargs):

        self.final_structure = self.structure

        if self.check_cover():
            self.post_processing()
            return self
        try:
            super().run(**kwargs)
            self.post_processing()
        except:

            if self.run_count <= 0:
                self.post_processing()

                return self
            error = re.compile(".*please rerun with smaller EDIFF, or copy CONTCAR.*")
            with open(self.run_dir.joinpath(f"vasp.out"), "r", encoding="utf8") as f:
                for line in f:
                    if error.match(line):
                        logging.info("复制CONTCAR继续优化。。。")
                        self.run_count -= 1
                        self.structure = Structure.from_file(self.run_dir.joinpath(f"CONTCAR"))
                        return self.run(**kwargs)
        return self

    def plot_energy_force(self):

        out = Outcar(self.run_dir.joinpath("OUTCAR"))

        out.read_pattern({
            "e_fr_energy": r"free  energy   TOTEN\s+=\s+([\d\-\.]+)",
        }, postprocess=float)

        energy = np.array(out.data["e_fr_energy"])

        energy = energy.flatten()

        a = out.read_table_pattern(r"TOTAL-FORCE \(eV/Angst\)\n\s*\-+\n", r"\s+".join([r"(\-*[\.\d]+)"] * 6), r"-*\n",
                                   last_one_only=False, postprocess=float)

        force = np.array(a)[:, :, 3:]
        force = force.reshape((force.shape[0], -1))
        max_froce = np.max(force, 1)

        result = np.vstack([np.arange(energy.shape[0]), energy, max_froce]).T

        fig, axes = plt.subplots(2, 1, sharex=True)
        axes1, axes2 = axes
        axes1.plot(result[:, 0], result[:, 1], label="energy", color="red")
        axes1.set_ylabel("energy(eV)")
        axes1.legend()

        axes2.plot(result[:, 0], result[:, 2], label="max force", color="green")
        axes2.set_ylabel("max force")
        axes2.legend()

        axes2.set_xlabel("steps")
        plt.tight_layout()
        plt.savefig(self.run_dir.joinpath("energy_forces.png"), dpi=150)


    def post_processing(self, result=None):
        if result is None:
            result = {}

        self.final_structure = Structure.from_file(self.run_dir.joinpath("CONTCAR"))
        self.final_structure.to(self.run_dir.parent.joinpath(
            f'{self.final_structure.composition.to_pretty_string()}-{self.function}.cif').as_posix())

        try:
            self.plot_energy_force()
        except:
            pass
class SCFJob(JobBase):
    def __init__(self, step_type="scf", **kwargs):

        super().__init__(step_type=step_type, **kwargs)
        """
        因为scf后面会用到很多 所以要根据job_type 区分不同场景的
        """

    @cached_property
    def incar(self):
        incar = super().incar
        if self.job_type in ["single_point_energy", "phono"]:
            incar["LWAVE"] = False
            incar["LCHARG"] = False
        return incar

    @cached_property
    def kpoints(self):
        """
        因为有的体系自洽是用的连续点模式
        重写一下
        :return:
        """

        if self.function in ["r2scan", "scan", "mbj"]:
            return BaseKpoints(self.kpoints_type).get_line_kpoints(self.path, self.function, self.structure)
        return super().kpoints

    def run(self, **kwargs):
        if self.check_cover():
            return self
        if self.function in ["hse", "gw", "r2scan", "scan", "mbj", "diag"]:
            if self.path.joinpath(f"pbe/{self.folder}").exists():
                cp_file(self.path.joinpath(f"pbe/{self.folder}/WAVECAR"), self.run_dir)
        return super().run(**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        """
        自洽的返回费米能级
        :return:
        """
        vasprun = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False, parse_dos=True)
        result[f"efermi_{self.function}"] = vasprun.efermi
        result[f"energy_{self.function}"] = vasprun.final_energy
        result[f"volume_{self.function}"] = vasprun.final_structure.volume

        if self.job_type == "single_point_energy":
            name = vasprun.final_structure.composition.to_pretty_string()
            config_type = self.structure.properties.get("Config_type", f"scf-{name}")

            write_to_xyz(self.run_dir.joinpath("vasprun.xml"), f"./result/{name}{GlobSuffix}.xyz", config_type,
                         append=True)

            write_to_xyz(self.run_dir.joinpath("vasprun.xml"), f"./result/train{GlobSuffix}.xyz", config_type,
                         append=True)


        return result


class WorkFunctionJob(SCFJob):
    def __init__(self, **kwargs):
        super().__init__(job_type="work_function", folder="work_function", step_type="scf", **kwargs)

    @cached_property
    def incar(self):
        incar = super().incar
        incar["LVHAR"] = True

        if get_vacuum_axis(self.structure, 10) is not None:
            incar["LDIPOL"] = True
            incar["IDIPOL"] = get_vacuum_axis(self.structure, 5) + 1


        return incar

    def post_processing(self, result=None):
        result = super().post_processing(result)
        loc = Locpot.from_file(self.run_dir.joinpath("LOCPOT"))

        fig = plt.figure()
        z_data = loc.get_average_along_axis(2)
        z_index = loc.get_axis_grid(2)

        plt.plot(z_index, z_data)
        plt.xlabel("Position (A)")
        plt.ylabel("Potential (eV)")
        plt.savefig(self.run_dir.joinpath(f"work_function_{self.function}.png"), dpi=self.dpi)
        np.savetxt(self.run_dir.joinpath(f"work_function_{self.function}.csv"), np.vstack([z_index, z_data]).T)

        vacuum_level = np.max(z_data)
        vasprun = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False, parse_dos=True)

        result[f"vacuum_level_{self.function}"] = vacuum_level
        result[f"work_function_{self.function}"] = vacuum_level - vasprun.efermi
        return result

class LobsterJob(JobBase):

    def __init__(self, **kwargs):

        super().__init__(step_type="scf", folder="cohp", **kwargs)

    def build_lobster(self, basis_setting):
        lobsterin_dict = {"basisSet": "pbeVaspFit2015", "COHPstartEnergy": -10.0, "COHPendEnergy": 10.0,
                          "cohpGenerator": "from 0.1 to 6.0 orbitalwise", "saveProjectionToFile": True}
        # every interaction with a distance of 6.0 is checked
        # the projection is saved
        if self.incar["ISMEAR"] == 0:
            lobsterin_dict["gaussianSmearingWidth"] = self.incar["SIGMA"]
        lobsterin_dict["skipdos"] = True
        lobsterin_dict["skipcoop"] = True
        lobsterin_dict["skipPopulationAnalysis"] = True
        lobsterin_dict["skipGrossPopulation"] = True
        # lobster-4.1.0
        lobsterin_dict["skipcobi"] = True
        lobsterin_dict["skipMadelungEnergy"] = True
        basis = [f"{key} {value}" for key, value in basis_setting.items()]
        lobsterin_dict["basisfunctions"] = basis

        self.lobster = Lobsterin(lobsterin_dict)
        self.lobster.write_lobsterin(self.run_dir.joinpath("lobsterin").as_posix())

        return self.lobster

    def run_lobster(self):
        logging.info("\t开始运行lobster")
        with cd(self.run_dir), open("lobster.out", "w") as f_std, open("lobster.err", "w", buffering=1) as f_err:
            subprocess.check_call(["lobster"], stdout=f_std, stderr=f_err, )
        logging.info("\tlobster分析结束")

    def run(self, **kwargs):
        if self.check_cover():
            return self

        return super().run(lobster=self.lobster, **kwargs)

    def extract_icohp(self):
        icohp = Icohplist(filename=self.run_dir.joinpath("ICOHPLIST.lobster").as_posix())

        icohps = icohp.icohpcollection

        elements_with_numbers = list(set(icohps._list_atom1 + icohps._list_atom2))

        def extract_number(element):
            match = re.search(r'(\d+)$', element)
            return int(match.group(1)) if match else None

        numbers = [extract_number(elem) for elem in elements_with_numbers]

        sorted_pairs = sorted(zip(elements_with_numbers, numbers), key=lambda x: x[1])
        sorted_elements_with_numbers = [pair[0] for pair in sorted_pairs]
        frame = pd.DataFrame(index=sorted_elements_with_numbers, columns=sorted_elements_with_numbers)
        for _icohp in icohps._icohplist.values():
            if _icohp._translation != [0, 0, 0]:
                continue
            frame.loc[_icohp._atom1, _icohp._atom2] = _icohp._icohp[Spin.up]
            if Spin.down in _icohp._icohp.keys():
                frame.loc[_icohp._atom2, _icohp._atom1] = _icohp._icohp[Spin.down]

        frame.to_csv(self.run_dir.joinpath("icohp.csv"))


    def post_processing(self, result=None):
        if result is None:
            result = {}

        lobsterout = Lobsterout(self.run_dir.joinpath("lobsterout").as_posix())
        result["basis"] = lobsterout.basis_functions
        result["charge_spilling"] = lobsterout.charge_spilling
        result["best_path"] = self.run_dir.as_posix()

        self.extract_icohp()



        return result


class DosJob(JobBase):

    def __init__(self, **kwargs):
        super().__init__(job_type="band_structure", step_type="dos", **kwargs)

    @cached_property
    def incar(self):
        incar = super().incar
        if self.function == "mbj":
            outcar = Outcar(self.path.joinpath("mbj/scf/OUTCAR").as_posix())
            outcar.read_pattern({"CMBJ": r'CMBJ =    (.*)'})
            if outcar.data["CMBJ"]:
                incar["CMBJ"] = outcar.data["CMBJ"][-1][0]
        return incar

    def run(self, **kwargs):

        if self.check_cover():
            return self
        cp_file(self.path.joinpath(f"{self.function}/scf/CHGCAR"), self.run_dir)
        cp_file(self.path.joinpath(f"{self.function}/scf/CHG"), self.run_dir)

        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)

        return super().run(**kwargs)

    def write_dos_file(self, path, data, headers):
        np.savetxt(self.run_dir.joinpath(path), data, delimiter=" ", fmt="%10.6f", comments="",
                   header=" ".join(headers))

    def export_tdos(self, tdos, dos):
        verify_path(self.run_dir.joinpath("data"))

        energy = dos.energies - dos.efermi

        self.write_dos_file("data/total-up.dat",
                            np.vstack([energy, tdos.densities[Spin.up]]).T,
                            headers=["energy(eV)", "Density"])
        if Spin.down in tdos.densities:
            self.write_dos_file("data/total-dw.dat",
                                np.vstack([energy, -tdos.densities[Spin.down]]).T,
                                headers=["energy(eV)", "Density"])

        # 先按元素导出所有元素的总的
        elem_dos = dos.get_element_dos()

        for elem, e_dos in elem_dos.items():

            self.write_dos_file(f"data/total-up-{elem.name}.dat",
                                np.vstack([energy, e_dos.densities[Spin.up]]).T,
                                headers=["energy(eV)", "Density"])
            if Spin.down in e_dos.densities:
                self.write_dos_file(f"data/total-dw-{elem.name}.dat",
                                    np.vstack([energy, -e_dos.densities[Spin.down]]).T,
                                    headers=["energy(eV)", "Density"])

    def export_pdos(self, dos: CompleteDos):
        verify_path(self.run_dir.joinpath("data/pdos"))

        energy = dos.energies - dos.efermi
        ispin = self.incar.get("ISPIN") == 2
        el_dos = {}
        index_map = {}

        for site, atom_dos in dos.pdos.items():
            element = site.specie.name
            if element not in el_dos:
                index_map[element] = 1
                el_dos[element] = {Spin.up: np.zeros((energy.shape[0], len(atom_dos)), dtype=np.float64)}
                if ispin:
                    el_dos[element][Spin.down] = el_dos[element][Spin.up].copy()

            site_single = {Spin.up: np.zeros_like(el_dos[element][Spin.up])}
            if ispin:
                site_single[Spin.down] = site_single[Spin.up].copy()

            for orb, pdos in atom_dos.items():
                for spin, ppdos in pdos.items():
                    site_single[spin][:, orb.value] += ppdos

            headers = ["energy(eV)"] + [Orbital(i).name for i in range(len(atom_dos))] + ["sum"]
            self.write_dos_file(
                f"data/pdos/site-up-{element}{index_map[element]}.dat",
                np.hstack(
                    [energy.reshape(-1, 1), site_single[Spin.up], site_single[Spin.up].sum(axis=1).reshape(-1, 1)]),
                headers
            )
            if ispin:
                self.write_dos_file(
                    f"data/pdos/site-dw-{element}{index_map[element]}.dat",
                    np.hstack([energy.reshape(-1, 1), -site_single[Spin.down],
                               -site_single[Spin.down].sum(axis=1).reshape(-1, 1)]),
                    headers
                )

            el_dos[element][Spin.up] += site_single[Spin.up]
            if ispin:
                el_dos[element][Spin.down] += site_single[Spin.down]

            index_map[element] += 1

        for elem, total_dos in el_dos.items():
            for spin, spin_dos in total_dos.items():
                headers = ["energy(eV)"] + [Orbital(i).name for i in range(spin_dos.shape[-1])] + ["sum"]

                self.write_dos_file(
                    f"data/pdos/element-{'up' if spin == Spin.up else 'dw'}-{elem}.dat",
                    np.hstack([energy.reshape(-1, 1), spin.value * spin_dos,
                               spin.value * spin_dos.sum(axis=1).reshape(-1, 1)]),
                    headers
                )



    def post_processing(self, result=None):
        if result is None:
            result = {}

        vasprun = Vasprun(self.run_dir.joinpath("vasprun.xml"), parse_potcar_file=False )
        dos = vasprun.complete_dos

        result[f"dos_efermi_{self.function}"] = dos.efermi
        result[f"dos_vbm_{self.function}"] = dos.get_cbm_vbm()[1]
        result[f"dos_cbm_{self.function}"] = dos.get_cbm_vbm()[0]
        result[f"dos_gap_{self.function}"] = dos.get_gap()

        self.export_tdos(vasprun.tdos, dos)
        if  setting.get("ExportProjection", True):
            self.export_pdos(dos)



        plotter = DosPlotter()
        # 添加各种元素的DOS数据
        for element, element_dos in dos.get_element_dos().items():

            if element_dos is not None:
                plotter.add_dos(str(element), element_dos)
        plot = plotter.get_plot(xlim=(self.vb_energy, self.cb_energy))
        # 将x轴的标签刻度只显示整数 注释掉会显示0.5这种
        # plot.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(self.run_dir.joinpath("dos.png"), dpi=self.dpi)
        return result


class BandStructureJob(JobBase):

    def __init__(self, **kwargs):
        super().__init__(job_type="band_structure", step_type="band", **kwargs)

    @cached_property
    def incar(self):
        incar = super().incar
        if self.function == "mbj":
            outcar = Outcar(self.path.joinpath("mbj/scf/OUTCAR").as_posix())
            outcar.read_pattern({"CMBJ": r'CMBJ =    (.*)'})
            if outcar.data["CMBJ"]:
                incar["CMBJ"] = outcar.data["CMBJ"][-1][0]

        return incar

    @cached_property
    def kpoints(self):
        """
        因为有的体系自洽是用的连续点模式
        重写一下
        :return:
        """

        if self.function in ["gw", "g0w0"]:
            return super().kpoints

        return BaseKpoints(self.kpoints_type).get_line_kpoints(self.path, self.function, self.structure)

    def run(self, **kwargs):
        if self.check_cover():
            return self
        cp_file(self.path.joinpath(f"{self.function}/scf/CHGCAR"), self.run_dir)
        cp_file(self.path.joinpath(f"{self.function}/scf/CHG"), self.run_dir)
        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)

        return super().run(**kwargs)

    def calculate_effective_mass(self, distance, energy, kpoint_index):
        window_size = 5

        # 计算窗口的一半大小
        half_window = window_size // 2

        # 确定窗口的左边界和右边界
        left_boundary = max(0, kpoint_index - half_window)
        right_boundary = min(len(energy), kpoint_index + half_window + (window_size % 2))  # 保证窗口大小为奇数

        # 如果窗口左边界在数组开头，调整右边界
        if left_boundary == 0:
            right_boundary = min(len(energy), right_boundary + (half_window - kpoint_index))

        # 如果窗口右边界在数组结尾，调整左边界
        if right_boundary == len(energy):
            right_boundary = min(len(energy), kpoint_index + half_window + (window_size % 2))
            left_boundary = max(0, right_boundary - window_size)

        energy *= 0.036749
        distance *= 0.5291772108
        coefficients = np.polyfit(distance[left_boundary:right_boundary], energy[left_boundary:right_boundary], 2)
        return 0.5 / coefficients[0]


    def write_projected_dat(self,filename,distance,energy,projected):

        _data = [energy]
        headers = [u"distance(1/A)", u"energy(eV)"]
        for i in range(projected.shape[-1]):
            _data.append(projected[:, :, i])
            headers.append(f"projected {Orbital(i).name}")
        headers.append("sum")

        _data.append(projected.sum(axis=2))
        array_to_dat(self.run_dir.joinpath(filename),
                     np.array(distance), *_data,
                     headers=headers)

    def export_projected_data(self,bs):
        verify_path(self.run_dir.joinpath("data/projected"))
        for spin, spin_projection in bs.projections.items():
            element_result = {}
            element_index={}
            for elem in self.structure.composition.elements:
                element_index[elem.name]=1
                element_result[elem.name] = np.zeros((spin_projection.shape[0], spin_projection.shape[1], spin_projection.shape[2]), dtype=np.float64)
            for site_index in range(len(self.structure)):
                site=self.structure[site_index].specie.name
                site_array=spin_projection[:,:, :, site_index]

                element_result[site]+=site_array
                self.write_projected_dat(
                    f"data/projected/site-{'up' if spin == Spin.up else 'dw'}-{site}{element_index[site]}.dat",
                    bs.distance,
                    bs.bands[spin]-bs.efermi,
                    site_array
                )
                element_index[site]+=1
            for elem ,value in element_result.items():

                self.write_projected_dat(
                    f"data/projected/element-{'up' if spin == Spin.up else 'dw'}-{elem}.dat",
                    bs.distance,
                    bs.bands[spin] - bs.efermi,
                    value
                )
    def export_band_data(self, bs):
        spin_map = {Spin.up: "up", Spin.down: "dw"}
        for spin, bands in bs.bands.items():
            spin_str = spin_map[spin]

            array_to_dat(self.run_dir.joinpath(f"data/band-{spin_str}.dat"), np.array(bs.distance), bands - bs.efermi,
                         headers=[u"distance(1/A)", u"energy(eV)"])

    def get_effective_mass(self, bs):
        me = 0
        mh = 0
        try:
            if not bs.is_metal():
                vbm = bs.get_vbm()
                cbm = bs.get_cbm()

                spin = list(cbm["band_index"].keys())[0]
                index = list(cbm["band_index"].values())[0][0]

                me = self.calculate_effective_mass(np.array(bs.distance),
                                                   bs.bands[spin][index].copy(),
                                                   cbm["kpoint_index"][0])

                spin = list(vbm["band_index"].keys())[0]
                index = list(vbm["band_index"].values())[0][0]

                mh = self.calculate_effective_mass(np.array(bs.distance),
                                                   bs.bands[spin][index].copy(),
                                                   vbm["kpoint_index"][0]
                                                   )

        except:
            pass
        return me, mh


    def post_processing(self, result=None):
        if result is None:
            result = {}
        if self.function != "pbe":

            force_hybrid_mode = True
        else:
            force_hybrid_mode = False

        if self.kpoints.style in [Kpoints.supported_modes.Line_mode, Kpoints.supported_modes.Reciprocal]:
            line_mode = True
        else:
            line_mode = False


        vasprun = BSVasprun(self.run_dir.joinpath("vasprun.xml").as_posix(),
                            parse_projected_eigen=setting.get("ExportProjection", True)
                            )

        bs = vasprun.get_band_structure(line_mode=line_mode, force_hybrid_mode=force_hybrid_mode)


        band_gap = bs.get_band_gap()
        vbm = bs.get_vbm()
        cbm = bs.get_cbm()
        result[f"direct_{self.function}"] = band_gap['direct']
        result[f"band_gap_{self.function}"] = band_gap['energy']
        result[f"vbm_{self.function}"] = vbm["energy"]
        result[f"cbm_{self.function}"] = cbm["energy"]
        result[f"efermi_{self.function}"] = bs.efermi
        result[f"m_e_{self.function}"], result[f"m_h_{self.function}"] = self.get_effective_mass(bs)



        if not line_mode:
            return result
        if not self.run_dir.joinpath("data").exists():
            self.run_dir.joinpath("data").mkdir()

        self.export_band_data(bs)
        if bs.projections:
            self.export_projected_data(bs)
        plotter = BSPlotter(bs)
        plot = plotter.get_plot(ylim=(self.vb_energy, self.cb_energy), vbm_cbm_marker=True)
        plt.savefig(self.run_dir.joinpath(f"band.png"), dpi=self.dpi)

        with open(self.run_dir.joinpath(f"data/band_lables.txt"), "w", encoding="utf8") as f:
            f.write("distance\tlable\n")
            distance = plotter.get_ticks()["distance"]
            label = plotter.get_ticks()["label"]
            for i in range(len(label)):
                f.write(f"{round(distance[i], 6)}\t{label[i]}\n")


        return result


class AimdJob(JobBase):

    def __init__(self, TEBEG=300, TEEND=300, NSW=3000, **kwargs):
        if "ML_LMLFF" in kwargs and kwargs["ML_LMLFF"]:
            folder = f"aimd-ml({TEBEG}-{TEEND}k)@{NSW}"
        else:
            folder = f"aimd({TEBEG}-{TEEND}k)@{NSW}"
        super().__init__(step_type="aimd", TEBEG=TEBEG, TEEND=TEEND, NSW=NSW, folder=folder, **kwargs)



    def run(self, **kwargs):
        if self.check_cover():
            return self

        return super().run(**kwargs)

    def plot_aimd(self, vasprun):

        name = vasprun.final_structure.composition.to_pretty_string()

        energies = [step["e_0_energy"] for step in vasprun.ionic_steps]
        steps = list(range(1, len(energies) + 1))
        plt.figure(figsize=(3.5, 2.625))
        plt.plot(steps, energies, label=name)
        plt.ylabel("E0 Energy(eV)")
        plt.xlabel("Time(fs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.run_dir.joinpath("aimd.png"), dpi=self.dpi)

    def get_ionic_steps_index(self, vasprun: Vasprun):
        index = 0
        result = []
        ionic_steps = vasprun.ionic_steps
        nionic_steps = vasprun.nionic_steps
        for md_i, md in enumerate(vasprun.md_data):

            if md["energy"]["e_0_energy"] == ionic_steps[index]["e_0_energy"]:
                result.append(md_i + 1)

                index += 1
                if index == nionic_steps:
                    break

        return result

    def plot_aimd_ml(self, vasprun):
        name = vasprun.final_structure.composition.to_pretty_string()

        energies = [step["energy"]["e_0_energy"] for step in vasprun.md_data]
        steps = list(range(1, len(energies) + 1))
        plt.figure()
        plt.plot(steps, energies, label=name)

        energies = [step["e_0_energy"] for step in vasprun.ionic_steps]

        index = self.get_ionic_steps_index(vasprun)
        if len(index) == len(energies):
            plt.scatter(index, energies, label="Aimd", s=4, c="red")

        plt.ylabel("E0 Energy(eV)")
        plt.xlabel("Time(fs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.run_dir.joinpath("aimd-ml.png"), dpi=self.dpi)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        """
        
        :return:
        """
        vasprun = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False, parse_dos=False)

        if self.incar.get("ML_LMLFF"):
            # 机器学习
            self.plot_aimd_ml(vasprun)
        else:
            self.plot_aimd(vasprun)
        config_type = f"{self.folder}-({self.path.name})-"

        write_to_xyz(self.run_dir.joinpath("vasprun.xml"), self.run_dir.joinpath("aimd.xyz"), config_type, append=False)
        return result


class StaticDielectricJob(JobBase):

    def __init__(self, **kwargs):
        super().__init__(job_type="optic_dielectric", step_type="dielectric", **kwargs)

    def run(self, **kwargs):
        if self.check_cover():
            return self

        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)
        return super().run(**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        outcar = Outcar(self.run_dir.joinpath("OUTCAR").as_posix())

        result[f"dielectric_electron_{self.function}"] = outcar.dielectric_tensor[0][0]
        if self.incar.get("IBRION") == 8:
            result[f"dielectric_ionic_{self.function}"] = outcar.dielectric_ionic_tensor[0][0]
        else:
            result[f"dielectric_ionic_{self.function}"] = 0

        return result


class ElasticJob(JobBase):

    def __init__(self, **kwargs):
        super().__init__(job_type="elastic", step_type="elastic", folder="elastic", **kwargs)

    # @cached_property
    # def incar(self):
    #     incar =super().incar
    #
    #
    #
    #     return incar
    def run(self, **kwargs):
        if self.check_cover():
            return self
        # cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)
        # cp_file(self.path.joinpath(f"{self.function}/scf/CHGCAR"), self.run_dir)

        return super().run(**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        outcar = Outcar(self.run_dir.joinpath("OUTCAR").as_posix())
        outcar.read_elastic_tensor()
        elastic_tensor = outcar.data["elastic_tensor"]
        result["elastic_tensor"] = elastic_tensor
        return result

class OpticJob(JobBase):
    result_label = ["dielectric_real", "dielectric_imag",
                    "optic_direct_band_gap", "optic_indirect_band_gap",
                    "mean", "max", "area"

                    ]

    def __init__(self, **kwargs):
        super().__init__(job_type="optic_dielectric", step_type="optic", **kwargs)

    @cached_property
    def incar(self):
        incar = super().incar
        if self.function in ["bse"]:

            incar["NBANDS"] = Wavecar(self.path.joinpath(f"gw/band/WAVECAR").as_posix()).nb
        else:
            eig = Eigenval(self.path.joinpath(f"{self.function}/scf/EIGENVAL").as_posix())
            incar["NBANDS"] = eig.nbands * 2
        return incar

    def run(self, **kwargs):
        if self.check_cover():
            return self
        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)
        return super().run(**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        vasp = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False)

        result[f"dielectric_real_{self.function}"] = vasp.dielectric[1][0][0]
        result[f"dielectric_imag_{self.function}"] = vasp.dielectric[2][0][0]

        new_en, new_abs = slme.absorption_coefficient(vasp.dielectric)

        plt.clf()
        plt.xlabel("Photon energy (eV)")
        plt.ylabel("Absorption ($cm^{-1}$)")
        plt.plot(new_en, new_abs)

        plt.xlim((0, 5))
        # plt.tight_layout()
        plt.savefig(self.run_dir.joinpath(f"absorption_coefficient.png"), dpi=self.dpi)

        info = {}
        for i, en in enumerate(new_en):
            if "start" not in info.keys():
                if en >= 1.59:
                    info["start"] = (i, en)
            if en >= 3.26:
                info["end"] = (i, en)
                break
        _max = round(np.max(new_abs[info["start"][0]:info["end"][0]]) / 1e6, 5)

        mean = round(np.mean(new_abs[info["start"][0]:info["end"][0]]) / 1e6, 5)
        result[f"mean_{self.function}"] = mean
        result[f"max_{self.function}"] = _max
        result[f"area_{self.function}"] = round(
            np.trapz(new_abs[info["start"][0]:info["end"][0]], new_en[info["start"][0]:info["end"][0]]) / 1e6, 5)

        plt.clf()

        plt.plot(vasp.dielectric[0], vasp.dielectric[1], label="real")
        plt.plot(vasp.dielectric[0], vasp.dielectric[2], label="imag")
        plt.ylim(-40, 40)
        plt.legend()
        plt.savefig(self.run_dir.joinpath(f"dielectric-function.png"), dpi=self.dpi)
        plt.clf()

        return result


class BaderJob(SCFJob):
    def __init__(self, **kwargs):
        super().__init__(job_type="bader", step_type="scf", folder="bader", **kwargs)

    @cached_property
    def incar(self):
        incar = super().incar
        incar["LAECHG"] = True

        return incar

    def save_summary(self, summary):
        with open(self.run_dir.joinpath("ACF.dat"), "w", encoding="utf8") as f:
            header = "Id,X,Y,Z,label,charge,transfer,min dist,atomic volume".split(",")

            header = [i.center(10) for i in header]
            header_text = "".join(header)
            f.write(header_text)
            f.write("\n")
            f.write("-" * 100)
            f.write("\n")

            for index in range(len(self.structure)):
                site = self.structure[index]
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

    def post_processing(self, result=None):
        result = super().post_processing(result)
        logging.info("\t开始bader电荷分析。")
        summary = bader_analysis_from_path(self.run_dir.as_posix())
        logging.info("\tbader电荷分析完成。")

        self.save_summary(summary)

        return result


@requires(Phonopy, "请先安装phonopy！")
class PhonopyJob():
    pass

    def __init__(self, structure: Structure, path: Path):
        self.structure = structure

        self.run_path = path.joinpath("pbe/phono")
        verify_path(self.run_path)
        self.ph_structure = phonopy.get_phonopy_structure(structure)
        self.phonon = Phonopy(unitcell=self.ph_structure, supercell_matrix=config["KPOINTS"]["phono"]["super"])
        self.phonon.generate_displacements(
            distance=0.01,
        )

        self.disp_supercells = self.phonon.supercells_with_displacements
        self.init_supercell = self.phonon.supercell
        logging.info(f"一共生成{len(self.disp_supercells)}个结构")
        displacements = self.phonon.displacements
        # write_disp_yaml(
        #     displacements=displacements,
        #     supercell=self.init_supercell,
        #     filename=self.path.joinpath("phonopy_disp.yaml"),
        # )
        units = get_default_physical_units("vasp")
        phpy_yaml = PhonopyYaml(
            physical_units=units, settings={}
        )
        phpy_yaml.set_phonon_info(self.phonon)
        with open(self.run_path.joinpath("phonopy_disp.yaml"), "w") as w:
            w.write(str(phpy_yaml))
        self.structure.to(self.run_path.joinpath("POSCAR").as_posix(), fmt="poscar")
        phonopy.get_pmg_structure(self.init_supercell).to(self.run_path.joinpath("SPOSCAR"), fmt="poscar")

    @property
    def supercell_structures(self):
        index = 1
        for cell in self.disp_supercells:
            if cell is not None:
                s = phonopy.get_pmg_structure(cell)
                s.to(self.run_path.joinpath(f"POSCAR-{index:03d}").as_posix(), fmt="poscar")
                index += 1
                yield s

    def set_forces(self, forces):
        self.phonon.forces = forces

        write_FORCE_SETS(self.phonon.dataset, self.run_path.joinpath("FORCE_SETS"))
        self.phonon.produce_force_constants(calculate_full_force_constants=False)
        write_FORCE_CONSTANTS(self.phonon.force_constants, filename=self.run_path.joinpath("FORCE_CONSTANTS"),
                              p2s_map=self.phonon.primitive.p2s_map)

    def get_bandstructure(self, plot=True):
        kpoint = BaseKpoints().get_line_kpoints(None, function="pbe", structure=self.structure, job_type="phono",
                                                step_type="band")
        labels_dict = {a: k for a, k in zip(kpoint.labels, kpoint.kpts) if a != ""}

        path = []
        labels = []
        for k, l in zip(kpoint.kpts, kpoint.labels):
            # 去除重复路径
            if path:
                if path[-1] == list(k):
                    continue
                else:
                    path.append(list(k))
            else:
                path.append(list(k))

            if l.strip():
                if labels:
                    if labels[-1] == l.strip():
                        continue
                    else:
                        labels.append(l.strip())
                else:
                    labels.append(l.strip())

        path = [path]

        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=kpoint.num_kpts,
                                                                     rec_lattice=self.structure.lattice.reciprocal_lattice.matrix)

        self.phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

        self.phonon.write_yaml_band_structure(None, filename=self.run_path.joinpath("band.yaml"))
        # 这里还没搞明白 感觉是pymatgen的PhononBSPlotter画图问题 先放下
        # qpoints = np.vstack(qpoints)
        # print(qpoints.shape)
        # self.phonon.run_qpoints(qpoints)
        # frequencies = self.phonon.band_structure.frequencies

        # frequencies = np.vstack(frequencies).T

        # frequencies = self.phonon.qpoints.frequencies.T
        # print(frequencies.shape)

        # phono_bandstructure=PhononBandStructureSymmLine(qpoints, frequencies, self.structure.lattice, labels_dict=labels_dict)
        if plot:
            self.phonon.plot_band_structure().savefig(self.run_path.joinpath("phonon_bandstructure.png"), dpi=150)
            # plotter = PhononBSPlotter(phono_bandstructure)
            # plotter.save_plot(self.run_path.joinpath("phonon_bandstructure.png"))
        # return phono_bandstructure


class VaspTool:
    def __init__(self, cores: int = None,
                 mpirun_path: Path = "mpirun",
                 vasp_path: Path = "vasp_std",
                 force_coverage: bool = False,
                 kpoints_type="Gamma",
                 functions: list = ["pbe"],
                 dft_u=False,
                 disable_relaxation=False,
                 open_soc=False,
                 incar_args={}
                 ):
        """

        :param cores: 指定运行的核数，如不指定，就默认使用本机最大核数
        :param mpirun_path: 如果没有设置环境变量，则需要设置下路径。
                            有环境变量默认即可
        :param vasp_path: 如果没有设置vasp环境变量，则需要设置下路径。
                            有环境变量默认即可
        :param force_coverage: 是否强制覆盖重复计算，
                                如果为False，计算前，如果存在4个输入文件以及OUTCAR，
                                    他们文件内容一致，就不再进行计算。
                                如果为True，则不检查文件，直接计算。
        :param functions:要使用的泛函方式 pbe  hse
        :param dft_u:是否开启加U
        :param disable_relaxation:禁止优化
        :param open_soc:使用vasp_ncl
        """
        if cores is None:
            cores = os.cpu_count()
        else:
            cores = cores

        self.mpirun_path = mpirun_path
        self.vasp_path = vasp_path
        self.functions = functions

        self.disable_relaxation = disable_relaxation
        self.fermi = 0
        # 这里是汇总下计算项添加的列

        self.check_params()
        self.job_args = {
            "dft_u": dft_u,
            "kpoints_type": kpoints_type,
            "open_soc": open_soc,
            "force_coverage": force_coverage,
            "mpirun_path": self.mpirun_path,
            "vasp_path": self.vasp_path,
            "cores": cores
        }
        self.incar_args = incar_args

    def check_params(self):
        """
        做一些自检 包括泛函选择、vasp路径等
        :return:
        """

        if not all(item in FUNCTION_TYPE for item in args.function):
            raise ValueError(f"function目前只支持{'、'.join(FUNCTION_TYPE)}")

        if not (self.vasp_path.exists()):
            vasp_std_path = get_command_path(self.vasp_path)
            if vasp_std_path:
                self.vasp_path = Path(vasp_std_path)
            else:
                raise ValueError(f"找不到文件：{self.vasp_path}")

        if not (self.mpirun_path.exists()):
            mpirun_path = get_command_path("mpirun")
            if mpirun_path:
                self.mpirun_path = Path(mpirun_path)
            else:
                raise ValueError(f"找不到文件：{self.mpirun_path}")

        logging.info("计算泛函为：" + "、".join(self.functions))
        logging.info(f"mpirun路径：{self.mpirun_path}")
        logging.info(f"VASP路径：{self.vasp_path}" + "\t开启soc后会自动切换到同目录下的ncl版本")

    def set_plot_setting(self, cbm: int, vbm: int, dpi: int):
        self.vb_energy = vbm
        self.cb_energy = cbm
        self.dpi = dpi

        self.job_args["vb_energy"] = vbm
        self.job_args["cb_energy"] = cbm
        self.job_args["dpi"] = dpi

    def plot_bs_dos(self, bs_path: Path, dos_path: Path, file_path: Path):
        """
        画出能带Dos组合图
        :param bs_path: 这里必须是具体到计算能带的vasprun.xml的路径
        :param dos_path: 这里必须是具体到计算dos的vasprun.xml的路径
        :param file_name: 要保存的图片路径,这里是路径。比如./band.png
        :return:
        """

        if not (os.path.exists(bs_path) and os.path.exists(dos_path)):
            logging.warning("必须计算完能带和dos后才能画能带dos图")
            return

        dos_vasprun = Vasprun(dos_path.as_posix(), parse_potcar_file=False)

        bs_vasprun = BSVasprun(bs_path.as_posix(), parse_projected_eigen=True)

        # 获取DOS数据
        dos = dos_vasprun.complete_dos

        bands = bs_vasprun.get_band_structure(line_mode=True)

        plotter = BSDOSPlotter(bs_projection="elements", dos_projection="orbitals",
                               vb_energy_range=-self.vb_energy, cb_energy_range=self.cb_energy, fixed_cb_energy=True,
                               fig_size=(8, 6))
        # 绘制DOS图
        plot = plotter.get_plot(bands, dos)
        plt.savefig(file_path, dpi=self.dpi)

    def count_optic_dielectric_by_gw_bse(self, structure_info: pd.Series, path: Path):
        band_job = BandStructureJob(structure=self.structure, path=path, function="gw", **self.job_args,
                                    **self.incar_args)

        band_job.run()
        band_job.post_processing(structure_info)

        optic_job = OpticJob(structure=self.structure, path=path, function="bse", **self.job_args, **self.incar_args)

        cp_file(band_job.run_dir.joinpath("WAVE*"), optic_job.run_dir)
        cp_file(band_job.run_dir.joinpath("*.tmp"), optic_job.run_dir)
        optic_job.run(remove_wavecar=True)
        optic_job.post_processing(structure_info)

        return structure_info

    def count_optic(self, structure_info: pd.Series, path: Path):

        self.structure = structure_info["structure"]
        # 进行结构优化
        # return self.count_optic_dielectric_by_gw_bse(structure_info,path)

        for function in self.functions:
            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="optic_dielectric", function=function,
                                             **self.job_args, **self.incar_args).run()
                self.structure = job.final_structure
            # # # 进行scf自洽计算
            scf_job = SCFJob(structure=self.structure, path=path,
                             job_type="optic_dielectric", function=function,
                             **self.job_args, **self.incar_args).run()

            scf_job.post_processing(structure_info)

            optic_job = OpticJob(structure=self.structure, path=path,
                                 function=function, **self.job_args, **self.incar_args).run(remove_wavecar=True)

            optic_job.post_processing(structure_info)

            structure_info[structure_info.index != 'structure'].to_csv(
                path.joinpath(f"{function}/result_{function}.csv"))

            structure_info["structure"] = self.structure

        return structure_info

    def count_dielectric(self, structure_info: pd.Series, path: Path):

        self.structure = structure_info["structure"]
        # 进行结构优化
        # return self.count_optic_dielectric_by_gw_bse(structure_info,path)

        for function in self.functions:
            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="optic_dielectric", function=function,
                                             **self.job_args, **self.incar_args).run()
                self.structure = job.final_structure
            # # # 进行scf自洽计算
            scf_job = SCFJob(structure=self.structure, path=path,
                             job_type="optic_dielectric", function=function,
                             **self.job_args, **self.incar_args).run()

            scf_job.post_processing(structure_info)

            # #进行介电常数的
            dielectric_job = StaticDielectricJob(structure=self.structure, path=path,
                                                 function=function, **self.job_args, **self.incar_args).run(
                remove_wavecar=True)

            dielectric_job.post_processing(structure_info)

            structure_info[structure_info.index != 'structure'].to_csv(
                path.joinpath(f"{function}/result_{function}.csv"))

            structure_info["structure"] = self.structure

        return structure_info

    def calculate_band_by_gw(self, path, function):

        band_job = BandStructureJob(structure=self.structure, path=path, function="gw", **self.job_args,
                                    **self.incar_args)

        band_job.run(remove_wavecar=True)
        result = band_job.post_processing()

    def count_band_structure(self, structure_info, path: Path = "./", channl="banddos") -> pd.Series:
        self.structure: Structure = structure_info["structure"]

        for function in self.functions:
            # # # 进行scf自洽计算

            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="band_structure", function=function,
                                             **self.job_args, **self.incar_args).run()

                self.structure = job.final_structure

            if function in ["gw"]:
                self.calculate_band_by_gw(path, function=function)

            scf_job = SCFJob(structure=self.structure, path=path,
                             job_type="band_structure", function=function,
                             **self.job_args, **self.incar_args).run()

            scf_job.post_processing(structure_info)

            if "dos" in channl:
                dos_job = DosJob(structure=self.structure, path=path,
                                 function=function, **self.job_args, **self.incar_args).run(remove_wavecar=True)

                dos_job.post_processing(structure_info)
                dos_vasprun = dos_job.run_dir.joinpath(f"vasprun.xml")
            else:
                dos_vasprun = path.joinpath(f"{function}/dos/vasprun.xm")
            if "band" in channl:
                band_job = BandStructureJob(structure=self.structure, path=path,
                                            function=function, **self.job_args, **self.incar_args).run(
                    remove_wavecar=True)

                band_job.post_processing(structure_info)
                band_vasprun = band_job.run_dir.joinpath(f"vasprun.xml")
            else:
                band_vasprun = path.joinpath(f"{function}/band/vasprun.xml")

            self.plot_bs_dos(band_vasprun, dos_vasprun,
                             path.joinpath(f"{function}/band_structure_dos_{function}.png"))
            structure_info[structure_info.index != 'structure'].to_csv(
                path.joinpath(f"{function}/result_{function}.csv"))
            structure_info["structure"] = self.structure

        return structure_info

    def count_cohp(self, structure_info, path: Path = "./"):
        self.structure: Structure = structure_info["structure"]

        if not self.disable_relaxation:
            job = StructureRelaxationJob(structure=self.structure,
                                         path=path,
                                         job_type="band_structure",
                                         function="pbe",
                                         **self.job_args, **self.incar_args
                                         ).run()

            self.structure = job.final_structure
        count = 1
        best_result = None
        all_possible_basis = Lobsterin.get_all_possible_basis_functions(self.structure,
                                                                        get_pot_symbols(self.structure.species))
        logging.info(f"可能的基组个数：{len(all_possible_basis)}")
        for basis_setting in all_possible_basis:
            # # # 进行scf自洽计算

            cohp_job = LobsterJob(
                test=count,
                structure=self.structure,
                path=path,
                job_type="cohp",
                function="pbe",
                **self.job_args, **self.incar_args
            )

            cohp_job.build_lobster(basis_setting)

            cohp_job.run()
            cohp_job.run_lobster()
            result = cohp_job.post_processing()
            result["basis"] = basis_setting

            if best_result is None:
                best_result = result
            else:
                if result["charge_spilling"] < best_result["charge_spilling"]:
                    best_result = result

            count += 1
        if best_result:
            for k, v in best_result.items():
                structure_info[k] = v

        structure_info[structure_info.index != 'structure'].to_csv(path.joinpath(f"pbe/cohp/result.csv"))

        return structure_info

    def count_aimd(self, structure_info, path: Path = "./"):

        self.structure: Structure = structure_info["structure"]
        if not self.disable_relaxation:
            job = StructureRelaxationJob(structure=self.structure, path=path,
                                         job_type="aimd", function="pbe",
                                         **self.job_args, **self.incar_args).run()

            self.structure = job.final_structure

        aimd_job = AimdJob(
            structure=self.structure, path=path,
            job_type="aimd", function="pbe",
            **self.job_args, **self.incar_args
        )
        aimd_job.run(remove_wavecar=True)
        aimd_job.post_processing(

        )
        return structure_info

    def count_elastic(self, structure_info, path: Path = "./"):

        self.structure: Structure = structure_info["structure"]
        if not self.disable_relaxation:
            job = StructureRelaxationJob(structure=self.structure, path=path,
                                         job_type="elastic", function="pbe",
                                         **self.job_args, **self.incar_args).run()

            self.structure = job.final_structure

        elastic_job = ElasticJob(
            structure=self.structure, path=path,
            function="pbe",
            **self.job_args, **self.incar_args
        )
        elastic_job.run(remove_wavecar=True)
        elastic_job.post_processing(structure_info

                                    )
        return structure_info
    def count_phono(self, structure_info, path: Path = "./"):
        self.structure: Structure = structure_info["structure"]
        pass

        self.incar_args["LREAL"] = False
        self.incar_args["PREC"] = "Accurate"

        if not self.disable_relaxation:
            job = StructureRelaxationJob(structure=self.structure, path=path,
                                         job_type="phono", function="pbe",
                                         **self.job_args, **self.incar_args).run()
            self.structure = job.final_structure

        phono_job = PhonopyJob(self.structure, path)
        forces = []
        for index, structure in enumerate(phono_job.supercell_structures):
            scf_job = SCFJob(structure=structure, path=path,
                             job_type="phono", function="pbe", test=index + 1,
                             **self.job_args, **self.incar_args).run(remove_wavecar=True)

            vasprun = Vasprun(scf_job.run_dir.joinpath("vasprun.xml"), parse_potcar_file=False)

            forces.append(vasprun.ionic_steps[0]["forces"])
        forces = np.array(forces)

        phono_job.set_forces(forces)

        result = phono_job.get_bandstructure(plot=True)

        return structure_info

    def count_scf(self, structure_info, path: Path = "./"):
        self.structure: Structure = structure_info["structure"]

        for function in self.functions:
            # # # 进行scf自洽计算

            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="single_point_energy", function=function,
                                             **self.job_args, **self.incar_args).run()
                self.structure = job.final_structure
            # 单点能暂时不考虑泛函了  如果后面考虑  需要考虑下波函数
            scf_job = SCFJob(structure=self.structure, path=path, folder="single_point_energy",
                             job_type="single_point_energy", function=function,
                             **self.job_args, **self.incar_args).run(remove_wavecar=True)

            scf_job.post_processing(structure_info)

        return structure_info

    def count_work_function(self, structure_info, path: Path = "./"):
        self.structure: Structure = structure_info["structure"]

        for function in self.functions:
            # # # 进行scf自洽计算

            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="work_function", function=function,
                                             **self.job_args, **self.incar_args).run()
                self.structure = job.final_structure
            # 这里考虑其他泛函的 比如hse 所以pbe的时候要输出一下自洽的波函数
            if len(self.functions) != 1:
                # 长度不等1  说明有其他泛函
                if function != "pbe":
                    remove_wavecar = True
                else:
                    remove_wavecar = False

            else:
                remove_wavecar = True
            scf_job = WorkFunctionJob(structure=self.structure, path=path,
                                      function=function,
                                      **self.job_args, **self.incar_args).run(remove_wavecar=remove_wavecar)

            scf_job.post_processing(structure_info)

        return structure_info

    def count_bader(self, structure_info, path: Path = "./"):
        self.structure: Structure = structure_info["structure"]

        for function in self.functions:
            # # # 进行scf自洽计算

            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="bader", function=function,
                                             **self.job_args, **self.incar_args).run()
                self.structure = job.final_structure

            scf_job = BaderJob(structure=self.structure, path=path,
                               function=function,
                               **self.job_args, **self.incar_args).run(remove_wavecar=True)

            scf_job.post_processing(structure_info)

        return structure_info

    def count_eos(self, structure_info, path: Path = "./"):
        self.structure: Structure = structure_info["structure"]
        step = config["SETTING"].get("EOSStep")
        step_num = config["SETTING"].get("EOSStepNum")
        step_num += step_num % 2
        for function in self.functions:
            # # # 进行scf自洽计算
            # self.structure.lattice.scale()
            if not self.disable_relaxation:
                job = StructureRelaxationJob(structure=self.structure, path=path,
                                             job_type="single_point_energy", function=function,
                                             **self.job_args, **self.incar_args).run()
                self.structure = job.final_structure

            start = round(-step * step_num / 2, 4)
            end = round(step * step_num / 2, 4)
            lattice = self.structure.lattice
            matrix = lattice.matrix.copy()
            lattice_map = {
                0: lattice.a,
                1: lattice.b,
                2: lattice.c
            }
            logging.info(f"搜索步长：{step} 搜索数量：{step_num}。晶格常数缩放范围：{start}-{end}")

            values = np.linspace(start, end, step_num + 1, dtype=float)
            values = np.around(values, 4)
            results = []
            for value in values:

                structure = self.structure.copy()
                if get_vacuum_axis(structure, 10) is None:
                    # 3维情况
                    for i, k in lattice_map.items():
                        matrix[i, :] = (matrix[i, :] / k) * (k + value)


                else:
                    for i, k in lattice_map.items():
                        if i == get_vacuum_axis(structure, 10):
                            continue
                        matrix[i, :] = (matrix[i, :] / k) * (k + value)
                structure.lattice = Lattice(matrix)

                scf_job = SCFJob(structure=structure, path=path, folder=f"eos/cache/{value}",
                                 job_type="single_point_energy", function=function,
                                 **self.job_args, **self.incar_args).run(remove_wavecar=True)
                result = scf_job.post_processing()
                result["index"] = value
                results.append(result)
            results = pd.DataFrame(results)

            eos = EOS(eos_name=config["SETTING"]["EOSModel"]).fit(results[f"volume_{function}"],
                                                                  results[f"energy_{function}"])
            eos.plot()
            plt.tight_layout()
            plt.savefig(path.joinpath(f"{function}/eos/eos.png"), dpi=self.job_args["dpi"])
            results.to_csv(path.joinpath(f"{function}/eos/eos.csv"))

            structure_info[f"e0_{function}"] = eos.e0
            structure_info[f"b0_{function}"] = eos.b0
            structure_info[f"b1_{function}"] = eos.b1
            structure_info[f"v0_{function}"] = eos.v0

        return structure_info

    def cb_sr(self, structure_info, path):
        self.structure: Structure = structure_info["structure"]
        job = StructureRelaxationJob(structure=self.structure, path=path,
                                     job_type="band_structure", function="pbe",
                                     **self.job_args, **self.incar_args).run()

        self.structure = job.final_structure
        return structure_info

    def test(self, structure_info, path):
        """
        k点测试demo
        通过传入KPOINTS给Job 自定义k点文件
        传入全大写的字段会默认给incar  比如SIGMA=5
        :param structure_info:
        :param path:
        :return:
        """
        self.structure: Structure = structure_info["structure"]
        result = []
        kps = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for i in kps:
            job = StructureRelaxationJob(structure=self.structure, path=path,
                                         job_type="band_structure", function="pbe", test=i,
                                         KPOINTS=Kpoints.gamma_automatic((i, i, i)), SIGMA=5,
                                         **self.job_args, **self.incar_args).run()
        final_energy = Outcar(job.run_dir.joinpath("OUTCAR")).final_fr_energy
        result.append(final_energy)
        plt.plot(kps, result)
        plt.savefig(job.run_dir.joinpath("test_kpoints.png"), dpi=self.dpi)
        return structure_info

    def count_main(self, file_path: Path, calculate_type="band"):

        structure_dataframe = read_dataframe_from_file(file_path)
        if structure_dataframe.empty:
            logging.error("计算为空，请检查输入文件")
            return

        logging.info(f"一共读取到{structure_dataframe.shape[0]}个文件")

        structure_dataframe: pd.DataFrame

        callback_function = {
            "band": partial(self.count_band_structure, channl="band"),
            "banddos": partial(self.count_band_structure, channl="banddos"),
            "dos": partial(self.count_band_structure, channl="dos"),

            # "band": self.count_band_structure,

            "optic": self.count_optic,
            "dielectric": self.count_dielectric,
            "elastic": self.count_elastic,
            "sr": self.cb_sr,
            "cohp": self.count_cohp,
            "test": self.test,
            "aimd": self.count_aimd,
            "aimd-ml": self.count_aimd,

            "phono": self.count_phono,
            "scf": self.count_scf,
            "work_function": self.count_work_function,
            "eos": self.count_eos,
            "bader": self.count_bader,

        }

        for index, struct_info in structure_dataframe.iterrows():
            try:
                if struct_info.get("calculate"):
                    continue
                path = Path(f"./cache/{struct_info['system']}{GlobSuffix}")

                if calculate_type in callback_function.keys():
                    struct_info = callback_function[calculate_type](struct_info, path)



            except KeyboardInterrupt:
                return

            except Exception:
                # 计算出错
                logging.error(traceback.format_exc())
                with open("./err.txt", "a+", encoding="utf8") as f:
                    f.write(struct_info['system'] + "\n")

            store_dataframe_as_json(struct_info.to_frame(), path.joinpath("result.json"))
            struct_info[struct_info.index != 'structure'].to_csv(path.joinpath("result.csv"))
            struct_info["calculate"] = True

            for i in struct_info.index:
                if i not in structure_dataframe.columns:
                    structure_dataframe.loc[:, i] = pd.NA

            structure_dataframe.loc[index] = struct_info

            if file_path.suffix == ".json":

                store_dataframe_as_json(structure_dataframe, file_path.name)
            else:
                store_dataframe_as_json(structure_dataframe, f"./result/all_result{GlobSuffix}.json")
                structure_dataframe.loc[:, structure_dataframe.columns != 'structure'].to_csv(
                    f"./result/result{GlobSuffix}.csv")

            # break
        logging.info("全部计算完成")


def build_argparse():
    parser = argparse.ArgumentParser(description="""Vasp计算脚本. 
    如果只计算pbe的带隙：python VaspTool.py band POSCAR
    如果计算hse能带：python VaspTool.py band POSCAR --function pbe hse
    计算杂化泛函以pbe为基础，所以hse前要加上pbe，泛函是按顺序执行的.""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "calculate_type", choices=calculate_type, help=f"要计算的类型，可以自己封装下，目前有:{'、'.join(calculate_type)}"
    )
    parser.add_argument(
        "path", type=Path, help="要计算的POSCAR路径，或者要批量计算的文件夹。"
    )
    parser.add_argument(
        "incar_args", type=str, help="对于INCAR的补充，将使用INCAR标准字段,可以设置多个空格隔开。例如 NSW=100 ENCUT=600",
        nargs="*"
    )

    parser.add_argument(
        "-v", "--version", action="version", version=__version__
    )

    group_vasp = parser.add_argument_group('计算细节', '设置K点类型、泛函等。')

    group_vasp.add_argument(
        "-k", "--kpoints_type", type=str, help="KPOINTS取点方式：Gamma、Monkhorst。可以只写首字母",
        default=setting.get("kpoints_type", "G")
    )

    group_vasp.add_argument(
        "--function", type=str, help="要使用的泛函方法比如pbe、hse", default=["pbe"], nargs="*"
    )

    group_vasp.add_argument(
        "-u", action='store_true', help="是否加U", default=False
    )
    group_vasp.add_argument(
        "-soc", "--open_soc", action='store_true', help="是否打开soc", default=False
    )
    group_vasp.add_argument(
        "--vdw", choices=list(config.get("VDW", {}).keys()), help="设置vdW 泛函", default=None
    )

    group_vasp.add_argument(
        "--disable_sr", action='store_true', help="是否禁止优化", default=False
    )

    group_run = parser.add_argument_group('任务相关', '设置计算核数、vasp、mpirun环境等。')
    group_run.add_argument(
        "-s", "--suffix", type=str, help="给文件夹名字以及输出文件添加一个后缀", default=setting.get("suffix", "")
    )
    group_run.add_argument(
        "-f", "--force_coverage", action='store_true', help="是否强制覆盖运行", default=False
    )
    group_run.add_argument(
        "-n", "-c", "--core", type=int, help="要计算使用的核数，默认为计算机最大核数。。", default=os.cpu_count()
    )
    group_run.add_argument(
        "--vasp_path", type=Path, help="vasp_std计算路径，如果设置环境变量，可以不传这个参数",
        default=setting.get("vasp_path", "G")
    )
    group_run.add_argument(
        "--mpirun_path", type=Path, help="mpirun 路径，如果设置环境变量，可以不传这个参数",
        default=setting.get("mpirun_path", "G")
    )
    group_plot = parser.add_argument_group('画图', '画图细节设置。')

    group_plot.add_argument(
        "--energy_min", type=int, help="画能带图的时候y轴的下限", default=setting.get("energy_min", "G")

    )
    group_plot.add_argument(
        "--energy_max", type=int, help="画能带图的时候y轴的上限", default=setting.get("energy_max", "G")

    )
    group_plot.add_argument(
        "--dpi", type=int, help="保存图的清晰度", default=setting.get("dpi", "G")

    )


    return parser


def parse_input_incar_value(input_values: list | None):
    result = {}

    if not input_values:
        return result
    for input_value in input_values:

        values = input_value.split("=")

        if len(values) != 2:
            logging.warning("输入的INCAR参数必须用等号连接，不同参数间用空格，比如：NSW=50。而不是：" + input_value)
            continue

        key, value = values
        try:
            v = Incar.proc_val(key, value)
        except:
            logging.warning("输入的INCAR参数必须用等号连接，不同参数间用空格，比如：NSW=50。而不是：" + input_value)
            continue
        result[key] = v
    logging.info(f"通过脚本传入的INCAR参数为：{result}")
    return result


if __name__ == '__main__':
    calculate_type = ["band", "dos", "banddos", "optic", "cohp",
                      "dielectric", "aimd", "aimd-ml", "phono", "elastic",
                      "scf", "work_function", "eos",
                      "bader"
                      ]
    parser = build_argparse()
    args = parser.parse_args()
    logging.info(f"任务使用核数：{args.core}")

    if not os.path.exists("./result"):
        os.mkdir("./result")
    incar_args = parse_input_incar_value(args.incar_args)
    if args.calculate_type == "aimd-ml":
        incar_args["ML_LMLFF"] = True
        incar_args["ML_MODE"] = "train"

    if args.vdw:
        vdw = config["VDW"][args.vdw]
        for k, v in vdw.items():
            incar_args[k] = v
        logging.info(f"设置VDW泛函{args.vdw}参数：{vdw}")

    vasp = VaspTool(vasp_path=args.vasp_path,
                    mpirun_path=args.mpirun_path,
                    force_coverage=args.force_coverage,
                    kpoints_type=args.kpoints_type,
                    cores=args.core,
                    functions=args.function,
                    dft_u=args.u,
                    disable_relaxation=args.disable_sr,
                    open_soc=args.open_soc,
                    incar_args=incar_args
                    )
    if args.suffix:
        # 多节点在同一路径计算 给每个job设置一个后缀 这样可以避免数据在同一个路径下计算造成数据覆盖
        GlobSuffix = f"-{args.suffix}"
    else:
        GlobSuffix = ""
    vasp.set_plot_setting(vbm=args.energy_min, cbm=args.energy_max, dpi=args.dpi)

    vasp.count_main(args.path, args.calculate_type)
