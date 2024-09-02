#! /bin/bash
#SBATCH --job-name=gpumd-python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-a800
#SBATCH --gres=gpu:1
module load gcc-10.1.0-gcc-10.1.0-3o3bvj2
source ~/.bashrc
 
export PATH=/opt/app/cuda/12.2/bin:${PATH}
export LD_LIBRARY_PATH=/opt/app/cuda/12.2/lib64:${LD_LIBRARY_PATH}
export PATH=~/opt/GPUMD-3.9.4/src:$PATH

__conda_setup="$('/opt/app/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/app/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/app/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/app/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate mysci
python GpumdTool.py learn  s -t 1000  -T {50..250..50} --template=nvt_nhc -s $SLURM_JOB_ID



