#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 23:31
# @Author  : 兵
# @email    : 1747193328@qq.com

import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
for i in path.iterdir():
    _path = i.as_posix()
    cmd = ["sbatch", "sub_vasp.sh", _path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
