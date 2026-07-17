#!/bin/bash
# FILENAME:  hugh

#SBATCH --job-name=hugh
#SBATCH --partition=pscomp,pscompl
#SBATCH --gres=gpu
#SBATCH --nodelist=h13
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH -o /data101/makinen/juplogs/hugh.o%j          # Name of stdout output file
#SBATCH -e /data101/makinen/juplogs/hugh.e%j          # Name of stderr error file
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# main script

port=1998
node=$(hostname -s)

#module load tensorflow/2.12
module load cuda/11.8
module load intelpython/3-2022.2.1
#module load cmake/3.25.1
#module load openmpi/4.1.2-gnu

XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

cd /home/makinen
source /home/makinen/venvs/fastjax/bin/activate

echo "Loading jupyter on node ${node} and port ${port}."

jupyter lab --no-browser --port=${port} --ip=${node}
