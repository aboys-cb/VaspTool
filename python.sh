#! /bin/bash
#SBATCH --job-name=ccb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
source /opt/intel/oneapi/setvars.sh
export PATH=~/opt/vasp.6.4.2/bin:$PATH

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
python -u VaspTool.py dielectric structures


