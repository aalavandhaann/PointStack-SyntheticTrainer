#!/bin/bash
#SBATCH --account=def-rnoumeir
#SBATCH --gres=gpu:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=20  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=02-15:30:00     # DD-HH:MM:SS
#SBATCH --mail-user=srinivasan.ramachandran.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL

#module restore nvccenvironment310 
module restore nvccenvironmentcp396
SOURCEDIR=/home/ashok82/projects/def-rnoumeir/ashok82/PointStack-SyntheticTrainer
# Prepare virtualenv
# source ~/Workspace/TensorFlowEnvironment/bin/activate
module load cuda/11.7
source ~/Workspace/TensorFlowEnvironment39/bin/activate

tensorboard --logdir=$SOURCEDIR/experiments/SyntheticPartNormal/synthetic_shapenet/tensorboard/ --host 0.0.0.0 --load_fast false & python $SOURCEDIR/train.py --cfg_file $SOURCEDIR/cfgs/syntheticpartnormal/syntheticpartnormal.yaml --exp_name synthetic_shapenet --val_steps=1
