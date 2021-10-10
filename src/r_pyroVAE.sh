#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N log_pyrovae_m0z2
#$ -cwd
#$ -l h_rt=6:00:00
#$ -l h_vmem=30G
#$ -pe gpu-titanx 2
#$ -R y

# Initialise PATH to see local scripts
PATH=$PATH:$HOME/.local/bin:$HOME/bin
export PATH

# Initialise the environment modules
. /etc/profile.d/modules.sh

#module load igmm/apps/R/3.6.1
#module load phys/compilers/gcc/9.1.0
#source /exports/applications/support/set_cuda_visible_devices.sh
#export LD_LIBRARY_PATH=/exports/applications/apps/SL7/cuda/10.1.105/extras/CUPTI/lib64/:/exports/applications/apps/SL7/cuda/10.1.105/:/exports/applications/apps/SL7/cuda/10.1.105/lib64
#export CUDA_HOME=/exports/applications/apps/SL7/cuda/10.1.105
#module load cuda/10.1.105

module load anaconda/5.3.1
source activate vae

# Run the program
#export HYDRA_FULL_ERROR=1
python main_pyroVAE.py
