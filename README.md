# VASPTool

## 介绍

VASPTool 主要用于计算能带、态密度、介电常数、光学性质等物理性质，支持一条命令自动完成计算。集成了多种泛函，并支持作业管理系统提交。

- 计算参数均是个人经验，如果不对请友好交流。
- 如果对计算参数有好的优化建议 欢迎交流改进。

### 功能特点

- 支持多种泛函：PBE、HSE、SCAN、R2SCAN、MBJ
- 支持单点能、能带、态密度、介电常数、光学、分子动力学计算
- 支持作业管理系统提交

### 社区支持

- [点击加入VASPTool群聊](https://qm.qq.com/q/wPDQYHMhyg)
- 通过issue提出问题和讨论交流

## 软件架构

建议使用 Python 3.10 以上版本。旧版本可能会报错类型错误。

### 目录结构

- `script`：一些画图脚本
- `VaspTool.py`：计算的主程序
- `config.yaml`：计算的主程序的配置文件

## 安装教程

在使用时 只需要复制
使用 pip 安装必需的库：`VaspTool.py`和`config.yaml`即可。

```bash
pip install pymatgen seekpath
```

如果需要处理 XYZ 文件，还需安装 ASE：

```bash
pip install ase
```

## 配置说明

- 修改 config.yaml 文件中的 `ExportXYZ` 设置为 True 以启用导出功能。
- 设置 `PMG_VASP_PSP_DIR` 为赝势文件路径。
- 如果 vasp_std 不在环境变量中，可在 config.yaml 修改 vasp_path 或通过命令行指定路径。

## 使用说明

查看帮助信息：

```bash
python VaspTool.py -h
```

### 计算示例

- 能带和态密度计算：

```bash
python VaspTool.py band Si2.cif
```

- 光学性质计算：

```bash
python VaspTool.py optic  Si2.cif
```

- 使用 HSE 泛函：

```bash
python VaspTool.py band Si2.cif --function pbe hse
```

### 分子动力学计算
会在VaspTool.py同级目录保存一个train.xyz的文件，所以输入文件不要叫train.xyz
- 计算单点能，禁止优化：

```bash
python VaspTool.py scf train1.xyz --disable_sr
```

- 计算单点能（先优化结构，在计算单点能）：

```bash
python VaspTool.py scf train1.xyz
```

- 分子动力学模拟

```bash
python VaspTool.py aimd POSCAR
```

## 参数调整

### INCAR参数

- 在config.yaml中指定元素的磁矩`MAGMOM`以及U值`U`，配置文件内置了一些从MP数据库获取的值。
- 对于INCAR基本参数 比如`ENCAT`,`EDIFF`等 可以修改config.yaml的`INCAR`.
- 对于特定性质的参数，一般默认即可，如果修改参考下面步骤。 比如分子动力学
    1. 在VaspTool.py搜索count_aimd 函数 大概在 1532 行。
    2. 找到下面的的代码
        ```python
        aimd_job=AimdJob(
             structure=self.structure, path=path,
             job_type="aimd", function="pbe",
             **self.job_args
         )
         ```
    3. 加入想要的参数 注意不要更改原有的东西，参数使用全大写的INCAR参数
         ```python
        aimd_job=AimdJob(
             structure=self.structure, path=path,
             job_type="aimd", function="pbe",
             TEBEG=600,TEEND=600,NSW=1000,
             **self.job_args
         )
         ```
    4. 其他Job同理 传入全大写字段即可。

### KPOINTS

- 在config.yaml的`KPOINTS`，可以指定每一步的k点文件，支持数字和列表（英文），比如3000和[8,8,8]
- 对于高对称路径的生成，使用seekpath生成，可能二维不是那么友好，可以自定义高对称路径
- 如果需要自定义高对称路径在脚本文件路径下定义一个HIGHPATH文件即可，文件内容为高对称路径。
- 对于杂化泛函的k点文件，如果存在pbe的计算路径，会默认使用vasp在pbe生成的k点文件，否则根据配置文件自动生成。
- 推荐在杂化泛函前加个pbe计算

### POTCAR

- 脚本全部使用VASP官网推荐的赝势文件，如果需要修改，在config.yaml的`POTCAR`

## 自定义教程
### K点测试
脚本封装了优化、自洽、以及一些非自洽计算。可以看具体每个类的注释。
对于k点测试，需要先定义一个函数 比如脚本中的test

```python
def test(self, structure_info, path ):

    self.structure = structure_info["structure"]
    result = []
    kps = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for i in kps:
        #调用优化的job
        job = StructureRelaxationJob(structure=self.structure, path=path,
                                     job_type="band_structure",function= "pbe",
                                     #上面四个参数是必须的 不用管
                                     test=i,#这里主要是为了建立文件夹
                                     KPOINTS=Kpoints.gamma_automatic((i,i,i)),#传入一个自定义的K
                                     **self.job_args).run()
    #取优化后的能量
    final_energy = Outcar(job.run_dir.joinpath( "OUTCAR")).final_fr_energy
    result.append(final_energy)
    plt.plot(kps, result)
    plt.savefig(job.run_dir.joinpath( "test_kpoints.png"), dpi=self.dpi)
    return structure_info
```
定义好执行函数后 在count_main的`callback_function`加入即可。





