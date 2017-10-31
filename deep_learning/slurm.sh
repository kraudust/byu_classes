#!/bin/bash
#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "insert_job_name_here"   # job name
#SBATCH --mail-user=dustan.kraus@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batchdj
# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
#%Module
module load defaultenv
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
module load python/2/7

python lab6_cancer_detector.py
