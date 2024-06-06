# VaspTool

#### 介绍
主要是为了方便计算Band DOS 光学等性质。可以一条命令自动计算
支持作业管理系统提交，
#### 软件架构
软件架构说明
最好python==3.10以上。3.8的提示有点问题，其他版本没有测试！
script是一些写的画图脚本。

#### 安装教程

1. `pip install pymatgen seekpath matminer`
2. 修改config.yaml中的PMG_VASP_PSP_DIR 这个是赝势路径
3. 如果vasp_std不在环境变量 可以在config.yaml 修改vasp_path
4. 点击链接加入群聊【VaspTool】：https://qm.qq.com/q/wPDQYHMhyg


#### 使用说明

1.  能带DOS：`python VaspTool band Si2.cif`
2. 光学：`python VaspTool optic Si2.cif`
2.  hse: `python VaspTool band Si2.cif --function pbe hse`

notes:如果需要自定义高对称路径在脚本文件路径下定义一个HIGHPATH文件即可，文件内容为高对称路径。
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





