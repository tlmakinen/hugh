#!/bin/bash
# FILENAME:  hugh

#SBATCH --job-name=hughrun
#SBATCH --partition=pscomp
#SBATCH --gres=gpu
#SBATCH --nodelist=j02
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH -o /data101/makinen/joblogs/hughrun.o%j          # Name of stdout output file
#SBATCH -e /data101/makinen/joblogs/hughrun.e%j          # Name of stderr error file
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# main script

module load cuda/11.8
module load intelpython/3-2022.2.1

XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

cd /home/makinen
source /home/makinen/venvs/fastjax/bin/activate

cd /home/makinen/repositories/hugh/torch_unet/

python train2.py --config configs_17_10.json


