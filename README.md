# VASPTool

[Github](https://github.com/aboys-cb/VaspTool) 国内可选择 [Gitee](https://gitee.com/QMaster/VaspTool)

## 介绍

VASPTool 主要用于计算能带、态密度、介电常数、光学性质等物理性质，支持一条命令自动完成计算。集成了多种泛函，并支持作业管理系统提交。

- 计算参数均是个人经验，如果不对请友好交流。
- 如果对计算参数有好的优化建议 欢迎交流改进。

### 功能特点

- 支持多种泛函：PBE、HSE、SCAN、R2SCAN、MBJ
- 支持声子谱（有限位移法）[phono]、单点能[scf]、能带态密度[banddos]、能带[band]、态密度[dos]、介电常数[dielectric]、光学[optic]
  、分子动力学计算[aimd]、分子动力学计算[aimd-ml]、功函数[work_function]、bader电荷[bader]
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

可以使用conda新建一个虚拟环境

```bash
conda create -n mysci python=3.10
```

使用 pip 安装必需的库：

```bash
pip install pymatgen seekpath
```
如果运行出现`ImportError`.需要降低numpy sci的版本，

```bash
pip install scipy==1.13.1
pip install numpy==1.26.1
```
如果需要处理 XYZ 文件，还需安装 ASE：

```bash
pip install ase
```

如果需要计算声子谱，还需安装 phonopy：

```bash
conda install -c conda-forge phonopy 
```

在使用时 只需要复制`VaspTool.py` 和`config.yaml`
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

最基本的提交格式

```bash
python VaspTool.py 计算类似 计算的模型路径[文件夹或者文件] 可选[一些incar设置 比如ISPIN=1]
```
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

- 进行声子谱计算：
- 这里传入的原胞，未扩包前的。扩包比例修改config.yaml的k点文件下KPOINTS->phono ->super(大概165行)

```bash
python VaspTool.py phono Si2.cif  EDIFF=1e-8 EDIFFG=-0.01 
```

### 分子动力学计算

会在VaspTool.py同级目录result保存一个train.xyz的文件
每个化合物也会保存一个单独的
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

# 作业管理系统提交(slurm为例)

## 单任务提交

使用路径下的python.sh即可

```bash
sbatch python.sh
```

## 多任务同时提交

- 首先将需要计算的结构分好类
- 假设现在我现在需要算500个结构的单点能 我想分成5个任务提交
- 我在脚本目录下新建一个`structures`的文件夹。
  - 如果我是每个结构单独的一个文件 比如：1.cif 2.cif
    - 那么请在`structures`新建5个文件夹（名字随意） 然后每个文件存放100个模型文件。
  - 如果我是一个文件包含很多模型的比如extxyz文件。
    - 那么将5个xyz文件放在`structures`即可
- 模型文件处理完成 开始准备提交
- 我们需要将`script/tool/sub_all.py` 上传到服务器 和`VaspTool.py`一个路径。
- 那么现在路径下有以下文件
  - VaspTool.py
  - config.yaml
  - python.sh
  - sub_all.py
  - structures
- 对于python.sh进行以下的修改 使其可以接受传入的参数。
  - 比如最后一行是：`python -u VaspTool.py dielectric structures`
  - 修改为`python -u VaspTool.py dielectric $1`
- 提交任务
  ```bash
  python sub_all.py  structures
  ```
## 参数调整

### INCAR参数

- 在config.yaml中指定元素的磁矩`MAGMOM`以及U值`U`，配置文件内置了一些从MP数据库获取的值。
- 对于INCAR基本参数 比如`ENCAT`,`EDIFF`等 可以修改config.yaml的`INCAR`.
  - 对于特定性质的参数，一般默认即可，如果修改参考下面步骤。 比如分子动力学
    - 命令行传参
    - 必须传入参数位置必须紧跟结构文件后面的位置 必须是INCAR标准字段 全大写。

      ```bash
      python VaspTool.py aimd POSCAR POTIM=1 NSW=5000 TEBEG=300 TEEND=500 --disable_sr
      ```
      其他Job同理 传入全大写字段即可。

### KPOINTS

- 在config.yaml的`KPOINTS`，可以指定每一步的k点文件，支持数字和列表（英文），比如3000和[8,8,8]
- 如果是整数 但是小于100 就会判定为按长度取k点。比如设置45 ，那么k就是45/a, 45/b,45/c。这个对于二维有用，有真空层的轴会自动设置1.
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





