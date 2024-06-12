#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 22:40
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

from monty.serialization import loadfn

__version__ = "1.0.2"

os.environ["PMG_DEFAULT_FUNCTIONAL"] = r"PBE_54"

config = loadfn("./config.yaml")
os.environ["PMG_VASP_PSP_DIR"] =  os.path.expanduser(os.path.expandvars(config["SETTING"]["PMG_VASP_PSP_DIR"]))




import abc
import argparse
import glob
import re
import shutil
import warnings
from pathlib import Path
import logging
import numpy as np
import json
import sys
import traceback

import pandas as pd
from monty.os import cd

import datetime
import os
import subprocess
from tqdm import tqdm
from monty.io import zopen
from monty.json import MontyEncoder, MontyDecoder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.lobster import Lobsterin,Lobsterout
from typing import *
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, VaspInput, Potcar, PotcarSingle
from pymatgen.io.vasp.outputs import Vasprun, BSVasprun, Outcar, Eigenval, Wavecar
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter, BSDOSPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.analysis.solar import slme
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from ase.io import read, write
except:
    write = None
    read = None
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rc('font', family='Times New Roman')

warnings.filterwarnings("ignore",module="pymatgen")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # 指定输出流为sys.stdout

)

PotcarSingle.functional_dir["PBE_54"] = ""
FUNCTION_TYPE = ["pbe","pbesol", "hse","scan","r2scan","mbj","gw","bse"]
KPOINTS_TYPE = Union[int, tuple,list]

potcar_config=config.get("POTCAR",{}).get("PBE54")



potcar_gw_config=config.get("POTCAR",{}).get("GW")


def hash_file(obj, file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = f.read()
    hash1 = hash(data)
    hash2 = hash(str(obj))
    return hash1 == hash2


def get_pot_symbols(species,mode:Literal["pbe54","gw"]="pbe54"):

    """
    根据传入 返回赝势列表
    :param species:
    :param mode:
    :return:
    """
    symbols = []
    for i in species:

        if mode=="pbe54":

            v = potcar_config[i.name]
        elif mode=="gw":

            v = potcar_gw_config[i.name]
        else:
            break
        if symbols:

            if symbols[-1]==v:
                continue
        symbols.append(v)

    return symbols
def cp_file(  source_file: Path, destination_dir: Path) -> None:
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
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


# 将xyz 获取的
def write_to_xyz(vaspxml_path, save_path, append=True):
    if setting.get("ExportXYZ"):

        if write is None:
            logging.error("设置开启了导出xyz文件，但没有安装ase，请 pip install ase")
        else:
            atoms_list = []
            atoms = read(vaspxml_path, index=":")
            for atom in atoms:
                xx, yy, zz, yz, xz, xy = -atom.calc.results['stress'] * atom.get_volume()  # *160.21766
                atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])

                atom.calc.results['energy'] = atom.calc.results['free_energy']

                atom.info['Config_type'] = "OUTCAR"
                atom.info['Weight'] = 1.0
                del atom.calc.results['stress']
                del atom.calc.results['free_energy']
                atoms_list.append(atom)

            write(save_path, atoms_list, format='extxyz', append=append)


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



def read_dataframe_from_file(file_path:Path, **kwargs) -> pd.DataFrame:
    """
    从指定路径读取结构 可以是文件夹路径、结构路径

    Returns: (pd.DataFrame)
    """
    if file_path.is_dir():

        systems=[]
        for p in  file_path.iterdir():

            try:
                s=read_dataframe_from_file(p)
                systems.append(s)
            except:
                logging.warning(f"读取结构文件{p}失败。")
                pass
        df = pd.concat(systems)

    else:

        if file_path.suffix.endswith(".json"):
            df =load_dataframe_from_json(file_path, **kwargs)
        elif file_path.name.endswith("POSCAR") or file_path.suffix in [".cif",".vasp"]:
            struct = Structure.from_file(file_path)

            if setting.get("UseInputFileName",False):
                system=file_path.stem
            else:
                system = struct.composition.formula.replace(" ", "")
            df = pd.DataFrame([{"system": system,
                                                 "structure": struct}])
        elif file_path.name.endswith("xyz"):
            systems = []
            if read is None:
                logging.error("xyz文件必须安装ase,请 pip install ase 安装！")
                return pd.DataFrame()
            atoms = read(file_path, index=":", format="extxyz")
            for atom in atoms:
                struct = AseAtomsAdaptor.get_structure(atom)
                #xyz 分子式一样 所以加个数字标识下
                systems.append({"system": struct.composition.formula.replace(" ","") ,
                                "structure": struct})
            df = pd.DataFrame(systems)

        else:
            raise ValueError(f"仅支持后缀为POSCAR、cif、vasp、json、xyz类型的文件")
    duplicated = df[df.duplicated("system", False)]

    group = duplicated.groupby("system")
    df["group_number"] = group.cumcount()
    df["group_number"].fillna(-1, inplace=True)
    df["group_number"] = df["group_number"].astype(int)
    df['system'] = df.apply(
        lambda row: f"{row['system']}-{row['group_number'] + 1}" if row['group_number'] >= 0 else row['system'], axis=1)
    df.drop("group_number", inplace=True, axis=1)
    df.reset_index(drop=True, inplace=True)
    return df
class BaseIncar(Incar):
    PBE_EDIFF=1e-06
    PBE_EDIFFG=-0.01
    HSE_EDIFF=1e-04
    HSE_EDIFFG=-0.01
    ENCUT=500
    def __init__(self,params:dict=None,**kwargs):
        super().__init__(params)

        self.update(kwargs)

    @classmethod
    def build(cls,system: str, function: FUNCTION_TYPE = "pbe",**kwargs)  :

        base=config.get("INCAR").copy()


        #不同泛函的基本参数
        # 关于杂化泛函 ICHARG=1比ICHARG=2快一倍  但是能量稍微差一点
        # Si2 hse dos
        # ICHARG=1: CBM:6.3352 VBM:5.3661  dos_gap:0.9691 耗费时间：30min
        # ICHARG=2: CBM:6.3218 VBM:5.3525  dos_gap:0.9693 耗费时间：12min
        if function == "hse":
            #因为HSE算的比较慢  所以降低精度
            base.update({
                "EDIFF": cls.HSE_EDIFF, "EDIFFG": cls.HSE_EDIFFG,
                "HFSCREEN": 0.2, "AEXX": 0.25, "LHFCALC": True,"PRECFOCK":"N"

            })
        elif function=="pbesol":
            base.update(GGA='PS')
        elif function=="scan":
            # "LDIAG": False 傻逼参数 目前体系加了回出现能带错乱
            base.update({"METAGGA": "SCAN", "ALGO": "ALL", "LASPH": True,
                         "LUSE_VDW": True, "BPARAM": 15.7, "CPARAM": 0.0093})

        elif function=="r2scan":
            base.update({"METAGGA": "R2SCAN",   "LASPH": True,
                         "LUSE_VDW": True, "BPARAM": 11.95, "CPARAM": 0.0093 })


        elif function=="mbj":
            base.update({"METAGGA": "MBJ",   "LASPH": True,
                           "GGA":"CA" ,"NELM":300 })


        elif function=="diag":
            base.update({"ALGO": "Exact", "LOPTICS": True,
                         "CSHIFT": 0.1, "NEDOS": 2000,"ISTART": 1})
            # system="diag"
            # base.pop("NPAR")

            #这里LPEAD=True 可以不加 看体系
        elif  function=="gw":
            base.update({"ALGO": "EVGW0", "LSPECTRAL": True, "NELMGW":1,
                         "ISTART": 1,"LOPTICS":True,"LREAL":False
                          })
            base.pop("NPAR")
        elif function=="bse":
            base.update({"ALGO": "BSE", "LSPECTRAL": True, "NELMGW": 1,
                         "ISTART": 1, "LOPTICS": True, "LREAL": False,
                         "NBANDSO":4,"NBANDSV":20,"OMEGAMAX":60
                         })
            base.pop("NPAR")

            system=""
        if function in ["scan","r2scan","mbj"]:
            if "GGA" in base.keys():
                base.pop("GGA")
        #---------------------------------------------------------------------
        if system=="sr":
            base.update({
                "LWAVE":False,"LCHARG":False,"NSW":30,"ISIF":3,"IBRION":2
            })
        elif  system=="scf":
            base.update({
                "LWAVE": True, "LCHARG": True, "NSW": 0, "IBRION": -1
            })
            if function=="hse":
                base.update({
                      "ISTART": 1,"ALGO":"Damped","ICHARG":0
                })
            elif function in [ "scan","r2scan"]:
                base.update({ "ALGO":"ALL","ICHARG": 2 })

        elif  system=="dos":
            base.update({
                "ISTART": 1, "ISMEAR": 0, "ICHARG": 11, "NSW": 0, "IBRION":-1,"LORBIT":11,
                "NEDOS":3000,"LWAVE":False,"LCHARG":False
            })

            if function=="hse":
                base.update({
              "ALGO":"Normal" ,"ICHARG":1,
                })
            elif function=="scan":
                base.update({  "ICHARG":1 ,
                              })
            elif function=="r2scan":
                base.update({ "ICHARG":1})
            elif  function=="mbj":
                base.update({"ICHARG": 2})
        elif  system=="band":
            base.update({
                "ISTART": 1, "ICHARG": 11, "NSW": 0, "IBRION":-1,"LORBIT":11, "LWAVE": False, "LCHARG": False
            })

            if function == "hse":
                base.update({
                   "ALGO": "Normal",
               "ICHARG": 1 ,
                })
            elif function=="scan":

                base.update({   "ICHARG":1 ,
                               })
            elif function=="r2scan":
                base.update({ "LREAL": False,
                              "ICHARG":1})
            elif  function=="mbj":
                base.update({"ICHARG": 1})

        elif system=="optic":
            base.update({
                "ISTART": 1, "NSW": 0,    "LWAVE": False,
                "LCHARG": False,"LOPTICS":True,"NBANDS":96,"NEDOS":2000,"CSHIF":0.100
            })
            if function == "hse":
                base.update({
                      "ICHARG": 2,"LREAL":False,"ALGO": "Normal"
                })
            elif function in ["pbesol","pbe"]:
                base.update({"IBRION": 8})

        elif system=="dielectric":
            base.update({
                "ISTART": 1,"SIGMA": 0.05 , "LEPSILON": True, "LPEAD": True, "IBRION": 8,"LWAVE":False,"LCHARG":False
            })
            base.pop("NPAR")
        elif system=="aimd":
            base.update({
                "ALGO": "N", "IBRION": 0, "MDALGO": 2, "ISYM": 0,
                "POTIM": 2, "NSW": 2000, "TEBEG": 300, "TEEND": 300,
                "SMASS": 1, "LREAL": "Auto", "ISIF": 2, "ADDGRID": True
            })

        base.update(kwargs)

        return cls(base)



    def has_magnetic(self, structure):
        """
        根据元素周期表判断体系是否具有磁性，如果有就打开自旋。
        :param system: 化学式，比如：CaTiO3
        :return: 返回(bool,str)
        """


        magmom = []
        spin = []
        _=[0,0]
        for site in structure.sites:
            if site.species_string in config.get("MAGMOM").keys()  :
                mag=config.get("MAGMOM")[site.species_string]
                spin.append(True)
            elif site.specie.name in config.get("MAGMOM").keys() :
                mag = config.get("MAGMOM")[site.specie.name ]
                spin.append(True)
            else:
                mag=0
                spin.append(False)
            if _[1]==mag:
                _[0]+=1
            else:
                magmom.append(f"{_[0]}*{_[1]}")
                _ = [1, mag]
        magmom.append(f"{_[0]}*{_[1]}")
        if any(spin):
            self["ISPIN"]=2
            self["MAGMOM"]=" ".join(magmom)



class BaseKpoints:
    _instance=None
    init_flag=False
    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self,kpoints_type):
        if BaseKpoints.init_flag:
            return

        BaseKpoints.init_flag = True
        self.kpoints_type=kpoints_type


        self.kpoints=config.get("KPOINTS")

    def get_kpoint_setting(self,job_type:str,step_type:str,function:str) :

        if job_type   not  in self.kpoints.keys():
            return 3000
        if step_type   not  in self.kpoints[job_type].keys():
            return 3000
        if function   not  in self.kpoints[job_type][step_type].keys():

            function="default"
        return self.kpoints[job_type][step_type][function]


    def get_kpoints(self,job_type:str,step_type:str,function:str,structure:Structure):
        kp =self.get_kpoint_setting(job_type, step_type, function)
        if isinstance(kp, int):
            kp = Kpoints.automatic_density( structure, kp).kpts[0]
        if self.kpoints_type.upper().startswith("M"):
            return Kpoints.monkhorst_automatic(kp)
        return Kpoints.gamma_automatic(kp)


    def get_line_kpoints(self,path:Path,function:str,structure:Structure) -> Kpoints:
        if function=="pbe":
            if os.path.exists("./HIGHPATH"):
                kpoints=Kpoints.from_file("./HIGHPATH")
            else:

                kpath = HighSymmKpath(structure, path_type="hinuma")
                kpoints = Kpoints.automatic_linemode(self.get_kpoint_setting("band_structure","band",function), kpath)
                # 下面这个循环 是将伽马点转换希腊字符，画图时的用
                for i, k in enumerate(kpoints.labels):
                    if k == "GAMMA":
                        kpoints.labels[i] = "\\Gamma"

            return kpoints

        if path.joinpath("pbe/band/vasprun.xml").exists():

            pbe_vasprun = BSVasprun(path.joinpath("pbe/band/vasprun.xml").as_posix())
            pbe_kpoints = Kpoints.from_file(path.joinpath("pbe/band/KPOINTS").as_posix())
            kpoints1 = Kpoints.from_file(path.joinpath("pbe/scf/IBZKPT").as_posix())

            kpoints = Kpoints("test", kpoints1.num_kpts + len(pbe_vasprun.actual_kpoints),
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
            kp = self.get_kpoint_setting("band_structure","scf",function)
            if isinstance(kp, int):
                grid = Kpoints.automatic_density(structure, kp).kpts[0]
            else:
                grid=kp
            ir_kpts = SpacegroupAnalyzer(structure, symprec=0.1).get_ir_reciprocal_mesh(grid )
            for k in ir_kpts:
                kpts.append(k[0])
                weights.append(int(k[1]))
                all_labels.append(None)

            # for line mode only, add the symmetry lines w/zero weight

            kpath = HighSymmKpath(structure, path_type="hinuma")
            frac_k_points, labels = kpath.get_kpoints(
                line_density=self.get_kpoint_setting("band_structure","band",function), coords_are_cartesian=False
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
    def __init__(self,structure,path,job_type,step_type,function,kpoints_type="Gamma",KPOINTS=None,open_soc=False,dft_u=False,force_coverage=False, mpirun_path="mpirun", vasp_path="vasp_std",cores=1,**kwargs):
        self.test=None
        self.structure=structure
        self.path:Path=path
        self.job_type=job_type
        self.step_type=step_type
        self.function=function
        self.open_soc=open_soc
        self.dft_u=dft_u
        self.kpoints_type=kpoints_type
        if KPOINTS is not None:
            assert isinstance(KPOINTS,Kpoints) ,f"自定义KPOINTS必须传入一个Kpoints对象而不是{type(KPOINTS)}"
        self.KPOINTS=KPOINTS
        self.force_coverage=force_coverage
        self.mpirun_path=mpirun_path
        self.vasp_path=vasp_path
        self.cores=cores
        self.cb_energy = 4
        self.dpi=300
        self.vb_energy = -4
        self.incar_kwargs={}
        for k,v in kwargs.items():
            if k.isupper():
                #暂且把全大写的分配到incar 后面有bug再说
                self.incar_kwargs[k]=v
            else:

                setattr(self, k, v)

        #要计算的类型 比如能带
        #要计算的类型的细分步骤 优化 自洽 性质等
        self.verify_path(self.run_dir)
        logging.info("当前计算路径："+self.run_dir.as_posix())
        if self.function in ["gw"]:
            self.pseudopotential="gw"
        else:
            self.pseudopotential="pbe54"

    @property
    def run_dir(self) -> Path:
        """
        获取vasp 计算路径
        :return:
        """
        if self.test is not None:
            return self.path.joinpath(f"{self.function}/{self.step_type}/{self.test}")

        return self.path.joinpath( f"{self.function}/{self.step_type}")

    @property
    def incar(self) ->BaseIncar:
        """Incar object."""


        incar = BaseIncar.build(self.step_type, self.function)
        formula = self.structure.composition.formula.replace(" ", "")
        incar["SYSTEM"] = formula + "-" + self.function + "-" + self.step_type
        incar.has_magnetic(self.structure)
        incar.update(self.incar_kwargs)
        if self.open_soc:
            incar["LSORBIT"] = True
        if self.dft_u and incar.get("LDAU") is None:
            data_u =config.get("U",{})

            if not data_u:
                logging.warning("\t开启DFT+U必须在配置文件设置U,开启失败!")
                return incar
            LDAUL = []
            LDAUU = []
            LDAUJ = []
            LDAUL_max=1
            for elem in self.structure.composition.elements:
                if elem.name in data_u.keys():
                    LDAUL.append(str(data_u[elem.name]["LDAUL"]))
                    LDAUU.append(str(data_u[elem.name]["LDAUU"]))
                    LDAUJ.append(str(data_u[elem.name]["LDAUJ"]))
                    if LDAUL_max<data_u[elem.name]["LDAUL"]:
                        LDAUL_max=data_u[elem.name]["LDAUL"]
                else:

                    LDAUL.append("-1")
                    LDAUU.append("0")
                    LDAUJ.append("0")


            if all([i =="-1" for i  in LDAUL]):
                logging.warning("\t在配置文件中没有找到该体系的U值,开启失败!")
                return incar
            incar["LDAU"] = True
            incar["LDAUTYPE"] = 2
            incar["LMAXMIX"] = LDAUL_max*2
            incar["LDAUL"] = " ".join(LDAUL)
            incar["LDAUU"] = " ".join(LDAUU)
            incar["LDAUJ"] = " ".join(LDAUJ)

        return incar
    @property
    def kpoints(self) -> Kpoints:
        """Kpoints object."""
        if self.KPOINTS is None:
            return BaseKpoints(self.kpoints_type).get_kpoints(self.job_type,self.step_type,self.function,self.structure)
        else:
            return self.KPOINTS


    @property
    def poscar(self) -> Poscar:
        """Poscar object."""
        poscar = Poscar(self.structure)
        return poscar
    @property
    def potcar(self) -> Potcar:
        potcar = Potcar(symbols=get_pot_symbols(self.structure.species,self.pseudopotential), functional="PBE_54")
        return potcar

    def verify_path(self, path: Path) -> None:
        """
        会检查是否存在路径，若不存在，则创建该路径，支持多级目录创建
        :param path:
        :return:
        """
        if not path.exists():
            # path.mkdir()
            os.makedirs(path)
    def check_cover(self ):
        """
        检查输入文件 避免重复计算 如果不需要重复计算 返回True 否则返回False
        :param run_dir:
        :return:
        """
        if not self.force_coverage and check_in_out_file(self.run_dir):
            hash_table = [
                hash_file(self.incar, self.run_dir.joinpath( "INCAR") ),
                hash_file(self.kpoints,  self.run_dir.joinpath( "KPOINTS") ),
                hash_file(self.poscar, self.run_dir.joinpath( "POSCAR") ),
                hash_file(self.potcar, self.run_dir.joinpath( "POTCAR") ),
            ]
            if all(hash_table):
                try:
                    if Outcar(os.path.join(self.run_dir, "OUTCAR")).run_stats.get("User time (sec)"):
                        logging.info("\t已有缓存，如果覆盖运行，设置force_coverage")

                        return True
                except:
                    pass
        src_files=["WAVE*","CHG*","*.tmp"]
        for src in src_files:
            src_file_list = glob.glob(self.run_dir.joinpath(src).as_posix())
            for file in src_file_list:
                Path(file).unlink()

        return False

    def run(self, timeout=None, lobster=None, remove_wavecar=False):
        if self.open_soc  :
            # 如果打开了soc 并且 scf  或band in
            vasp_path =  self.vasp_path.with_name("vasp_ncl")
        else:
            vasp_path = self.vasp_path
        vasp_input = VaspInput(self.incar, self.kpoints, self.poscar, self.potcar)
        vasp_cmd = [ self.mpirun_path, "-np", str( self.cores), vasp_path]

        start = datetime.datetime.now()
        logging.info("\t开始计算"  )
        vasp_input.write_input(output_dir=self.run_dir)
        if lobster:

            lobster.write_INCAR(incar_input=self.run_dir.joinpath("INCAR"),incar_output=self.run_dir.joinpath("INCAR"),poscar_input=self.run_dir.joinpath("POSCAR"))
        vasp_cmd = vasp_cmd or SETTINGS.get("PMG_VASP_EXE")  # type: ignore[assignment]
        if not vasp_cmd:
            raise ValueError("No VASP executable specified!")
        vasp_cmd = [os.path.expanduser(os.path.expandvars(t)) for t in vasp_cmd]
        if not vasp_cmd:
            raise RuntimeError("You need to supply vasp_cmd or set the PMG_VASP_EXE in .pmgrc.yaml to run VASP.")
        with cd(self.run_dir), open("vasp.out", "w") as f_std, open("vasp.err", "w", buffering=1) as f_err:
            subprocess.check_call(vasp_cmd, stdout=f_std, stderr=f_err, timeout=timeout)
        logging.info("\t计算完成"  +f"\t耗时：{datetime.datetime.now() - start}")
        if remove_wavecar:
            self.run_dir.joinpath("WAVECAR").unlink()

        return self
    @abc.abstractmethod
    def post_processing(self, result=None):
        if result is None:
            result = {}
class StructureRelaxationJob(JobBase):
    """
    结构优化的类
    """
    def __init__(self, **kwargs):
        super().__init__(step_type="sr",**kwargs )
        #vasp 有时会让复制contcar 继续优化  这个是控制复制次数
        self.run_count = 3
    def run(self,**kwargs):

        self.final_structure = self.structure

        if self.check_cover():
            self.post_processing()
            return self
        try:
            super().run( **kwargs )
            self.post_processing()
        except:

            if self.run_count<=0:
                self.post_processing()

                return self
            error = re.compile(".*please rerun with smaller EDIFF, or copy CONTCAR.*")
            with open(self.run_dir.joinpath(f"vasp.out"), "r", encoding="utf8") as f:
                for line in f:
                    if error.match(line):
                        logging.info("复制CONTCAR继续优化。。。")
                        self.run_count-=1
                        self.structure=Structure.from_file(self.run_dir.joinpath(f"CONTCAR"))
                        return self.run(**kwargs )
        return self
    def post_processing(self, result=None):
        if result is None:
            result = {}

        self.final_structure = Structure.from_file(self.run_dir.joinpath("CONTCAR"))
        self.final_structure.to(self.run_dir.parent.joinpath(f'{self.structure.composition.formula.replace(" ","")}-{self.function}.cif').as_posix())



class SCFJob(JobBase):


    def __init__(self,  **kwargs):
        super().__init__( step_type="scf", **kwargs)
        """
        因为scf后面会用到很多 所以要根据job_type 区分不同场景的
        """

    @property
    def incar(self):
        incar=super().incar
        if self.function in [ "diag"]:
            eig = Eigenval(self.path.joinpath("pbe/scf/EIGENVAL").as_posix())
            incar["NBANDS"] =eig.nbands*10


        return incar

    @property
    def kpoints(self):
        """
        因为有的体系自洽是用的连续点模式
        重写一下
        :return:
        """

        if self.function in ["r2scan","scan","mbj"]:
            return BaseKpoints(self.kpoints_type).get_line_kpoints(self.path,self.function,self.structure)
        return super().kpoints

    def run(self,**kwargs):
        if self.check_cover():
            return self
        if self.function in ["hse", "gw", "r2scan", "scan", "mbj", "diag"]:
            if self.path.joinpath("pbe/scf").exists():
                cp_file(self.path.joinpath("pbe/scf/WAVECAR"),  self.run_dir)
        return super().run(**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        """
        自洽的返回费米能级
        :return:
        """
        vasprun = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False, parse_dos=False)
        result[f"efermi_{self.function}"]=vasprun.efermi
        result[f"energy_{self.function}"]=vasprun.final_energy
        if self.job_type == "single_point_energy":
            write_to_xyz(self.run_dir.joinpath("vasprun.xml"), "./train.xyz", append=True)

        return result


class LobsterJob(JobBase):

    def __init__(self,basis,  **kwargs):
        self.basis=basis

        super().__init__( step_type="scf", **kwargs)

    @property
    def run_dir(self) -> Path:
        return self.path.joinpath(f"{self.function}/cohp/{self.basis}")


    def build_lobster(self,basis_setting):
        lobsterin_dict = {"basisSet": "pbeVaspFit2015", "COHPstartEnergy": -10.0, "COHPendEnergy": 5.0,
                          "cohpGenerator": "from 0.1 to 6.0 orbitalwise", "saveProjectionToFile": True}
        # every interaction with a distance of 6.0 is checked
        # the projection is saved
        if self.incar["ISMEAR"] == 0:
            lobsterin_dict["gaussianSmearingWidth"] = self.incar["SIGMA"]
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

    def run(self,**kwargs):
        if self.check_cover():
            return self


        return super().run(lobster=self.lobster,**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}

        lobsterout=Lobsterout(self.run_dir.joinpath("lobsterout").as_posix())
        result["basis"]=lobsterout.basis_functions
        result["charge_spilling"] =lobsterout.charge_spilling
        result["best_path"]=self.run_dir
        return result
class DosJob(JobBase):

    def __init__(self,  **kwargs):
        super().__init__(job_type="band_structure",step_type= "dos", **kwargs)


    @property
    def incar(self):
        incar=super().incar
        if self.function=="mbj":
            outcar=Outcar(self.path.joinpath("mbj/scf/OUTCAR").as_posix())
            outcar.read_pattern({"CMBJ": r'CMBJ =    (.*)'})
            if outcar.data["CMBJ"]:
                incar["CMBJ"]=outcar.data["CMBJ"][-1][0]
        return incar
    def run(self,**kwargs):

        if self.check_cover():
            return self
        cp_file(self.path.joinpath(f"{self.function}/scf/CHGCAR"),  self.run_dir)
        cp_file(self.path.joinpath(f"{self.function}/scf/CHG"),  self.run_dir)

        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)

        return super().run( **kwargs  )
    def post_processing(self, result=None):
        if result is None:
            result = {}

        vasprun = Vasprun(self.run_dir.joinpath("vasprun.xml"), parse_potcar_file=False)
        dos = vasprun.complete_dos
        result[f"dos_efermi_{self.function}"] = dos.efermi
        result[f"dos_vbm_{self.function}"] = dos.get_cbm_vbm()[1]
        result[f"dos_cbm_{self.function}"] = dos.get_cbm_vbm()[0]
        result[f"dos_gap_{self.function}"] = dos.get_gap()

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

    def __init__(self,  **kwargs):
        super().__init__(job_type="band_structure", step_type="band", **kwargs)

    @property
    def incar(self):
        incar=super().incar
        if self.function=="mbj":
            outcar=Outcar(self.path.joinpath("mbj/scf/OUTCAR").as_posix())
            outcar.read_pattern({"CMBJ": r'CMBJ =    (.*)'})
            if outcar.data["CMBJ"]:
                incar["CMBJ"]=outcar.data["CMBJ"][-1][0]

        return incar

    @property
    def kpoints(self):
        """
        因为有的体系自洽是用的连续点模式
        重写一下
        :return:
        """

        if self.function in ["gw","g0w0" ]:
            return super().kpoints

        return BaseKpoints(self.kpoints_type).get_line_kpoints(self.path,self.function,self.structure)


    def run(self,**kwargs):
        if self.check_cover():
            return self
        cp_file(self.path.joinpath(f"{self.function}/scf/CHGCAR"), self.run_dir )
        cp_file(self.path.joinpath(f"{self.function}/scf/CHG"), self.run_dir)
        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)

        return super().run(**kwargs  )

    def calculate_effective_mass(self,distance, energy, kpoint_index):
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

    def post_processing(self, result=None):
        if result is None:
            result = {}
        if self.function != "pbe":

            force_hybrid_mode = True
        else:
            force_hybrid_mode = False

        if self.kpoints.style in [Kpoints.supported_modes.Line_mode,Kpoints.supported_modes.Reciprocal]:
            line_mode=True
        else:
            line_mode=False

        vasprun = BSVasprun(self.run_dir.joinpath( "vasprun.xml").as_posix())


        bs = vasprun.get_band_structure(line_mode=line_mode, force_hybrid_mode=force_hybrid_mode)


        band_gap = bs.get_band_gap()
        vbm=bs.get_vbm()
        cbm=bs.get_cbm()
        result[f"direct_{self.function}"] = band_gap['direct']
        result[f"band_gap_{self.function}"] = band_gap['energy']
        result[f"vbm_{self.function}"] = vbm["energy"]
        result[f"cbm_{self.function}"] = cbm["energy"]
        result[f"efermi_{self.function}"] = vasprun.efermi
        try:
            if not bs.is_metal():

                spin = list(cbm["band_index"].keys())[0]
                index=list(cbm["band_index"].values())[0][0]

                result[f"m_e_{self.function}"] =self.calculate_effective_mass(np.array(bs.distance),
                                                                              bs.bands[spin][index],
                                                                              cbm["kpoint_index"][0]
                                                                              )

                spin = list(vbm["band_index"].keys())[0]
                index = list(vbm["band_index"].values())[0][0]

                result[f"m_h_{self.function}"] = self.calculate_effective_mass(np.array(bs.distance),
                                                                               bs.bands[spin][index],
                                                                               vbm["kpoint_index"][0]
                                                                               )

        except:
            pass
        if not line_mode:
            return result
        for spin, bands in bs.bands.items():
            np.savetxt(self.run_dir.joinpath(f"band{spin}.csv"),
                       np.vstack((np.array(bs.distance), bands - vasprun.efermi)).T, delimiter=",", fmt='%f')
        plotter = BSPlotter(bs)
        plot = plotter.get_plot(ylim=(self.vb_energy, self.cb_energy), vbm_cbm_marker=True)


        with open(self.run_dir.joinpath(f"band_lables.txt"), "w", encoding="utf8") as f:
            f.write("distance\tlable\n")
            distance = plotter.get_ticks()["distance"]
            label = plotter.get_ticks()["label"]
            for i in range(len(label)):
                f.write(f"{round(distance[i], 6)}\t{label[i]}\n")

        plt.savefig(self.run_dir.joinpath(f"band.png"), dpi=self.dpi)


        return result


class AimdJob(JobBase):


    def __init__(self,  **kwargs):
        super().__init__( step_type="aimd", **kwargs)


    # @property
    # def incar(self):
    #     incar=super().incar

    #
    #     return incar



    def run(self,**kwargs):
        if self.check_cover():
            return self


        return super().run(**kwargs)

    def post_processing(self, result=None):
        if result is None:
            result = {}
        """
        
        :return:
        """
        # vasprun = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False, parse_dos=False)
        # result[f"efermi_{self.function}"]=vasprun.efermi
        # result[f"energy_{self.function}"]=vasprun.final_energy
        write_to_xyz(self.run_dir.joinpath("vasprun.xml"), self.run_dir.joinpath("train.xyz"), append=False)
        return result

class  StaticDielectricJob(JobBase):

    def __init__(self,  **kwargs):
        super().__init__(job_type="optic_dielectric",step_type= "dielectric", **kwargs)

    def run(self,**kwargs ):
        if self.check_cover():
            return self


        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"), self.run_dir)
        return super().run( **kwargs )

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

class  OpticJob(JobBase):
    result_label = ["dielectric_real", "dielectric_imag",
                    "optic_direct_band_gap", "optic_indirect_band_gap",
                    "mean", "max", "area"

                    ]
    def __init__(self,  **kwargs):
        super().__init__(job_type="optic_dielectric",step_type= "optic", **kwargs)


    @property
    def incar(self):
        incar=super().incar
        if self.function   in ["bse"]:

            incar["NBANDS"] =Wavecar(self.path.joinpath(f"gw/band/WAVECAR").as_posix()).nb
        else:
            eig = Eigenval(self.path.joinpath(f"{self.function}/scf/EIGENVAL").as_posix())
            incar["NBANDS"] = eig.nbands*2
        return incar
    def run(self,**kwargs ):
        if self.check_cover():
            return self
        cp_file(self.path.joinpath(f"{self.function}/scf/WAVECAR"),self.run_dir)
        return super().run(**kwargs  )
    def post_processing(self, result=None):
        if result is None:
            result = {}
        vasp = Vasprun(self.run_dir.joinpath(f"vasprun.xml"), parse_potcar_file=False)

        result[f"dielectric_real_{self.function}"] = vasp.dielectric[1][0][0]
        result[f"dielectric_imag_{self.function}"] = vasp.dielectric[2][0][0]

        new_en,new_abs = slme.absorption_coefficient(vasp.dielectric)

        plt.clf()
        plt.plot(new_en, new_abs)


        plt.xlim((0, 5))
        # plt.tight_layout()
        plt.savefig(self.run_dir.joinpath(f"absorption_coefficient.png"),dpi=self.dpi)




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
        plt.ylim(-40,40)
        plt.legend()
        plt.savefig(self.run_dir.joinpath(f"dielectric-function.png"),dpi=self.dpi)
        plt.clf()

        return result
class VaspTool:
    def __init__(self, cores: int = None,
                 mpirun_path: Path = "mpirun",
                 vasp_path: Path = "vasp_std",
                 force_coverage: bool = False,
                 kpoints_type="Gamma",
                 functions:list=["pbe"],
                 dft_u=False,
                 disable_relaxation=False,
                 open_soc=False
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
        self.functions=functions


        self.disable_relaxation=disable_relaxation
        self.fermi = 0
        # 这里是汇总下计算项添加的列


        self.check_params()
        self.job_args={
            "dft_u":dft_u,
            "kpoints_type":kpoints_type,
            "open_soc":open_soc,
            "force_coverage":force_coverage,
            "mpirun_path":self.mpirun_path,
            "vasp_path":self.vasp_path,
            "cores":cores
        }
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
                self.vasp_path=Path(vasp_std_path)
            else:
                raise ValueError(f"找不到文件：{self.vasp_path}")

        if not (self.mpirun_path.exists()):
            mpirun_path = get_command_path("mpirun")
            if mpirun_path:
                self.mpirun_path = Path(mpirun_path)
            else:
                raise ValueError(f"找不到文件：{self.mpirun_path}")



        logging.info("计算泛函为："+"、".join(self.functions))
        logging.info(f"mpirun路径：{self.mpirun_path}")
        logging.info(f"VASP路径：{self.vasp_path}"+"开启soc后会自动切换到同目录下的ncl版本")

    def set_plot_setting(self,cbm:int,vbm:int,dpi:int):
        self.vb_energy=vbm
        self.cb_energy= cbm
        self.dpi=dpi

        self.job_args["vb_energy"]=vbm
        self.job_args["cb_energy"]=cbm
        self.job_args["dpi"]=dpi







    def plot_bs_dos(self, bs_path: Path, dos_path: Path, file_path: Path):
        """
        画出能带Dos组合图
        :param bs_path: 这里必须是具体到计算能带的vasprun.xml的路径
        :param dos_path: 这里必须是具体到计算dos的vasprun.xml的路径
        :param file_name: 要保存的图片路径,这里是路径。比如./band.png
        :return:
        """
        if not ( os.path.exists(bs_path) and os.path.exists(dos_path)):
            logging.warning("必须计算完能带和dos后才能画能带dos图")
            return

        dos_vasprun = Vasprun(dos_path.as_posix(),parse_potcar_file=False)

        bs_vasprun = BSVasprun(bs_path.as_posix(),parse_projected_eigen=True)

        # 获取DOS数据
        dos = dos_vasprun.complete_dos

        bands = bs_vasprun.get_band_structure(line_mode=True)


        # from pymatgen.electronic_structure.plotter import DosPlotter
        plotter = BSDOSPlotter( bs_projection="elements",dos_projection="orbitals",
                                vb_energy_range=-self.vb_energy,cb_energy_range=self.cb_energy,fixed_cb_energy=True,fig_size=(8,6))
        # 绘制DOS图
        plot = plotter.get_plot(bands, dos)
        plt.savefig(file_path, dpi=self.dpi)



    def count_optic_dielectric_by_gw_bse(self,structure_info: pd.Series, path:Path):
        band_job = BandStructureJob(structure=self.structure, path=path,function= "gw", **self.job_args)

        band_job.run()
        band_job.post_processing(structure_info)

        optic_job = OpticJob(structure=self.structure, path=path,function="bse", **self.job_args)

        cp_file(band_job.run_dir.joinpath("WAVE*"),optic_job.run_dir)
        cp_file(band_job.run_dir.joinpath("*.tmp"),optic_job.run_dir)
        optic_job.run(remove_wavecar=True)
        optic_job.post_processing(structure_info)



        return structure_info
    def count_optic(self, structure_info: pd.Series, path:Path):

        self.structure = structure_info["structure"]
        # 进行结构优化
        # return self.count_optic_dielectric_by_gw_bse(structure_info,path)

        for function in self.functions:
            if not self.disable_relaxation:
                job=StructureRelaxationJob(structure=self.structure, path=path,
                                           job_type="optic_dielectric",   function=function,
                                           **self.job_args).run( )
                self.structure=job.final_structure
            # # # 进行scf自洽计算
            scf_job=SCFJob(structure=self.structure, path=path,
                           job_type="optic_dielectric",   function=function,
                           **self.job_args).run()

            scf_job.post_processing(structure_info)



            optic_job = OpticJob(structure=self.structure, path=path,
                                 function=function, **self.job_args).run(remove_wavecar=True)

            optic_job.post_processing(structure_info)

            structure_info[structure_info.index != 'structure'].to_csv(path.joinpath(f"{function}/result_{function}.csv"))

            structure_info["structure"]=self.structure


        return structure_info

    def count_dielectric(self, structure_info: pd.Series, path:Path):

        self.structure = structure_info["structure"]
        # 进行结构优化
        # return self.count_optic_dielectric_by_gw_bse(structure_info,path)

        for function in self.functions:
            if not self.disable_relaxation:
                job=StructureRelaxationJob(structure=self.structure, path=path,
                                           job_type="optic_dielectric",   function=function,
                                           **self.job_args).run( )
                self.structure=job.final_structure
            # # # 进行scf自洽计算
            scf_job=SCFJob(structure=self.structure, path=path,
                           job_type="optic_dielectric",   function=function,
                           **self.job_args).run()

            scf_job.post_processing(structure_info)

            # #进行介电常数的
            dielectric_job = StaticDielectricJob(structure=self.structure, path=path,
                                                 function=function, **self.job_args).run(remove_wavecar=True)

            dielectric_job.post_processing(structure_info)


            structure_info[structure_info.index != 'structure'].to_csv(path.joinpath(f"{function}/result_{function}.csv"))

            structure_info["structure"]=self.structure


        return structure_info


    def calculate_band_by_gw(self,path,function):

        band_job = BandStructureJob(structure=self.structure, path=path, function="gw", **self.job_args)

        band_job.run(remove_wavecar=True)
        result = band_job.post_processing()

    def count_band_structure(self, structure_info , path:Path="./") ->pd.Series:
        self.structure :Structure= structure_info["structure"]


        for function in self.functions:
            # # # 进行scf自洽计算

            if not self.disable_relaxation:
                job=StructureRelaxationJob(structure=self.structure, path=path,
                                           job_type="band_structure",   function=function,
                                           **self.job_args).run( )


                self.structure=job.final_structure

            if function in ["gw"]:
                self.calculate_band_by_gw(path, function=function)

            scf_job=SCFJob(structure=self.structure, path=path,
                           job_type="band_structure",   function=function,
                           **self.job_args).run()

            scf_job.post_processing(structure_info)


            dos_job = DosJob(structure=self.structure, path=path,
                             function=function, **self.job_args).run(remove_wavecar=True)

            dos_job.post_processing(structure_info)
            band_job = BandStructureJob(structure=self.structure, path=path,
                                        function=function, **self.job_args).run(remove_wavecar=True)

            band_job.post_processing(structure_info)
            self.plot_bs_dos(band_job.run_dir.joinpath(f"vasprun.xml"),dos_job.run_dir.joinpath(f"vasprun.xml"),path.joinpath(f"{function}/band_structure_dos_{function}.png"))
            structure_info[structure_info.index != 'structure'].to_csv(path.joinpath(f"{function}/result_{function}.csv"))
            structure_info["structure"]=self.structure

        return structure_info

    def count_cohp(self, structure_info, path:Path="./"):
        self.structure :Structure= structure_info["structure"]

        if not self.disable_relaxation:
            job = StructureRelaxationJob(structure=self.structure,
                                         path=path,
                                         job_type="band_structure",
                                         function="pbe",
                                         **self.job_args
                                         ).run()

            self.structure = job.final_structure
        count=1
        best_result=None

        for basis_setting in Lobsterin.get_all_possible_basis_functions(self.structure,
                                                                  get_pot_symbols(self.structure.species)):
            # # # 进行scf自洽计算

            cohp_job=LobsterJob(
                basis=count,
                structure=self.structure,
                path=path,
                job_type="cohp",
                function="pbe",
                **self.job_args
                           )

            cohp_job.build_lobster(basis_setting)


            cohp_job.run()
            cohp_job.run_lobster()
            result=cohp_job.post_processing()
            result["basis"]=basis_setting


            if best_result is None:
                best_result=result
            else:
                if result["charge_spilling"] < best_result["charge_spilling"]:
                    best_result=result

            count+=1
        for k,v in best_result:
            structure_info[k]=v

        structure_info[structure_info.index != 'structure'].to_csv(path.joinpath(f"/pbe/cohp/result.csv"))

        return structure_info

    def count_aimd(self, structure_info, path:Path="./"):

        self.structure: Structure = structure_info["structure"]
        if not self.disable_relaxation:

            job = StructureRelaxationJob(structure=self.structure, path=path,
                                         job_type="aimd", function="pbe",
                                         **self.job_args).run()

            self.structure = job.final_structure

        aimd_job=AimdJob(
            structure=self.structure, path=path,
            job_type="aimd", function="pbe",
            **self.job_args
        )
        aimd_job.run(remove_wavecar=True)
        aimd_job.post_processing(

        )
        return structure_info

    def count_scf(self, structure_info, path:Path="./"):
        self.structure :Structure= structure_info["structure"]

        for function in self.functions:
            # # # 进行scf自洽计算

            if not self.disable_relaxation:
                job=StructureRelaxationJob(structure=self.structure, path=path,
                                           job_type="single_point_energy",   function=function,
                                           **self.job_args).run( )

            scf_job = SCFJob(structure=self.structure, path=path,
                             job_type="single_point_energy", function=function,
                             **self.job_args).run(remove_wavecar=True)

            scf_job.post_processing(structure_info)

        return structure_info
    def cb_sr(self, structure_info, path ):
        self.structure :Structure= structure_info["structure"]
        job = StructureRelaxationJob(structure=self.structure, path=path,
                                     job_type="band_structure",function= "pbe",
                                     **self.job_args).run()

        self.structure = job.final_structure
        return structure_info

    def test(self, structure_info, path ):
        """
        k点测试demo
        通过传入KPOINTS给Job 自定义k点文件
        传入全大写的字段会默认给incar  比如SIGMA=5
        :param structure_info:
        :param path:
        :return:
        """
        self.structure :Structure= structure_info["structure"]
        result = []
        kps = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for i in kps:
            job = StructureRelaxationJob(structure=self.structure, path=path,
                                         job_type="band_structure",function= "pbe",test=i,KPOINTS=Kpoints.gamma_automatic((i,i,i)),SIGMA=5,
                                         **self.job_args).run()
        final_energy = Outcar(job.run_dir.joinpath( "OUTCAR")).final_fr_energy
        result.append(final_energy)
        plt.plot(kps, result)
        plt.savefig(job.run_dir.joinpath( "test_kpoints.png"), dpi=self.dpi)
        return structure_info

    def count_main(self, file_path:Path, calculate_type="band"):


        structure_dataframe = read_dataframe_from_file(file_path)
        if structure_dataframe.empty:
            logging.error("计算为空，请检查输入文件")
            return
        logging.info(f"一共读取到{structure_dataframe.shape[0]}个文件")

        structure_dataframe: pd.DataFrame
        callback_function={
            "band":self.count_band_structure,
            "optic": self.count_optic,
            "dielectric": self.count_dielectric,
            "sr": self.cb_sr,
            "cohp": self.count_cohp,
            "test": self.test,
            "aimd":self.count_aimd,
            "scf": self.count_scf

        }

        for index, struct_info in structure_dataframe.iterrows():
            try:
                if struct_info.get("calculate"):
                    continue
                path=Path(f"./cache/{struct_info['system']}")

                if calculate_type in callback_function.keys():

                    struct_info = callback_function[calculate_type](struct_info, path)



            except KeyboardInterrupt:
                    return

            except Exception:
                # 计算出错
                logging.error(traceback.format_exc())
                with open("./err.txt", "a+", encoding="utf8") as f:
                    f.write(struct_info['system'] + "\n")

            store_dataframe_as_json(struct_info.to_frame(), f"./cache/{struct_info['system']}/result.json")
            struct_info[struct_info.index != 'structure'].to_csv(f"./cache/{struct_info['system']}/result.csv")
            struct_info["calculate"] = True

            for i in struct_info.index:
                if i not  in structure_dataframe.columns:
                    structure_dataframe.loc[:, i] = 0

            structure_dataframe.loc[index] = struct_info
            if file_path.suffix==".json":

                store_dataframe_as_json(structure_dataframe, file_path.name)
            else:
                store_dataframe_as_json(structure_dataframe, "./all_result.json")
                structure_dataframe.loc[:, structure_dataframe.columns != 'structure'].to_csv(f"./result.csv")

            # break
        logging.info("全部计算完成")

def build_argparse():

    parser = argparse.ArgumentParser(description="""Vasp计算脚本. 
    如果只计算pbe的带隙：python VaspTool.py band POSCAR
    如果计算hse能带：python VaspTool.py band POSCAR --function pbe hse
    计算杂化泛函以pbe为基础，所以hse前要加上pbe，泛函是按顺序执行的.""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "calculate_type",choices=calculate_type,help=f"要计算的类型，可以自己封装下，目前有:{'、'.join(calculate_type)}"
    )
    parser.add_argument(
        "path",type=Path,help="要计算的POSCAR路径，或者要批量计算的文件夹。"
    )


    parser.add_argument(
        "-v","--version",action="version",version=__version__
    )

    group_vasp = parser.add_argument_group('计算细节', '设置K点类型、泛函等。')

    group_vasp.add_argument(
        "-k","--kpoints_type",type=str,help="KPOINTS取点方式：Gamma、Monkhorst。可以只写首字母",default=setting.get("kpoints_type","G")
    )

    group_vasp.add_argument(
        "--function",type=str,help="要使用的泛函方法比如pbe、hse",default=["pbe"],nargs="*"
    )

    group_vasp.add_argument(
        "-u",action='store_true' ,help="是否加U",default=False
    )
    group_vasp.add_argument(
        "-soc","--open_soc",action='store_true' ,help="是否打开soc",default=False
    )

    group_vasp.add_argument(
         "--disable_sr",action='store_true',help="是否禁止优化",default=False
    )

    group_run = parser.add_argument_group('任务相关', '设置计算核数、vasp、mpirun环境等。')
    group_run.add_argument(
        "-f","--force_coverage",action='store_true',help="是否强制覆盖运行",default=False
    )
    group_run.add_argument(
        "-n","-c","--core",type=int,help="要计算使用的核数，默认为计算机最大核数。。",default=os.cpu_count()
    )
    group_run.add_argument(
        "--vasp_path",type=Path,help="vasp_std计算路径，如果设置环境变量，可以不传这个参数",default=setting.get("vasp_path","G")
    )
    group_run.add_argument(
        "--mpirun_path",type=Path,help="mpirun 路径，如果设置环境变量，可以不传这个参数",default=setting.get("mpirun_path","G")
    )
    group_plot = parser.add_argument_group('画图', '画图细节设置。')

    group_plot.add_argument(
        "--energy_min" ,type=int,help="画能带图的时候y轴的下限",default=setting.get("energy_min","G")

    )
    group_plot.add_argument(
        "--energy_max", type=int, help="画能带图的时候y轴的上限", default=setting.get("energy_max","G")

    )
    group_plot.add_argument(
        "--dpi", type=int, help="保存图的清晰度", default=setting.get("dpi","G")

    )

    return parser
if __name__ == '__main__':
    setting=config.get("SETTING",{})
    calculate_type=["band","optic","cohp","dielectric","aimd","scf"]
    parser=build_argparse()
    args=parser.parse_args()

    vasp = VaspTool(vasp_path=args.vasp_path,
                    mpirun_path=args.mpirun_path,
                    force_coverage=args.force_coverage,
                    kpoints_type=args.kpoints_type,
                    cores=args.core,
                    functions=args.function,
                    dft_u=args.u,
                    disable_relaxation=args.disable_sr,
                    open_soc=args.open_soc
                    )
    vasp.set_plot_setting(vbm=args.energy_min,cbm=args.energy_max,dpi=args.dpi)

    vasp.count_main(args.path,args.calculate_type)
