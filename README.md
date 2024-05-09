# VaspTool

#### 介绍
主要是为了方便计算Band DOS 光学等性质。可以一条命令自动计算
支持作业管理系统提交，
#### 软件架构
软件架构说明
最好python==3.10以上。3.8的提示有点问题，其他版本没有测试！
script是一些写的画图脚本。

#### 安装教程

1. pip install pymatgen seekpath matminer
2. 修改config.yaml中的PMG_VASP_PSP_DIR 这个是赝势路径
3. 如果vasp_std不在环境变量 可以在config.yaml 修改vasp_path
4. 点击链接加入群聊【VaspTool】：https://qm.qq.com/q/wPDQYHMhyg


#### 使用说明

1.  能带DOS：python VaspTool band Si2.cif
2. 光学：python VaspTool optic Si2.cif
2.  hse: python VaspTool band Si2.cif --function pbe hse





