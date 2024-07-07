#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 14:32
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
等间距抽取多少个结构在xyz文件中
每隔50抽取1个
python extract_xyz.py aimd.xyz 50

主动学习的 可以用 100k是标记 在Config_type
python extract_xyz.py dump.xyz 50 100k

"""
import argparse

import tqdm
from ase.io import read, write


def extract(file_path, num, config=None):
    atoms_info = {}
    atoms = read(file_path, index=":", format="extxyz")

    extract = atoms[::num]
    if config is not None:
        for i, atom in tqdm.tqdm(enumerate(extract), total=len(extract)):
            symbols = str(atom.symbols)

            if symbols not in atoms_info.keys():
                atoms_info[symbols] = 1
            atom.info["Config_type"] = f"{symbols}-{config}-{atoms_info[symbols]}"
            atoms_info[symbols] += 1
    print(f"抽取到{len(extract)}个结构。")
    # 这里将抽取的追加写入到微扰的里面
    write(f"./extract_{file_path}.xyz", extract, format='extxyz', append=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="等间距抽取指定文件中的结构。")
    parser.add_argument("filename", help="源数据文件", type=str)
    parser.add_argument("step", help="每隔step抽取一个结构", type=int)
    parser.add_argument("-c", "--config", help="生成Config_type.不指定则使用源数据文件的，如果没有则为空。", default=None,
                        type=str)
    args = parser.parse_args()

    extract(args.filename, args.step, args.config)
