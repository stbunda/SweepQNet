#!/bin/bash
# parameters for slurm
#SBATCH -J sweepnet                   # job name, don't use spaces, keep it short
#SBATCH -c 8                          # number of cores, 1
#SBATCH --gres=gpu:1                  # number of gpus 1, some clusters don't have GPUs
#SBATCH --mail-type=BEGIN,FAIL,END          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=15-00:00                # time limit 1h
#SBATCH --output=slurm_logs/sweepnet_train_%j.log      # Standard output and error log
#SBATCH -p main,dmb
#SBATCH --constraint=a40
 
# load all required software modules
module load nvidia/nvhpc/23.3	
module load anaconda3/2022.05
source activate
conda activate sweepnet
 
# It's nice to have some information logged for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Hostname -i       = $(hostname -I)"
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : "$SLURM_NNODES
echo "Number of MPI ranks         : "$SLURM_NTASKS
echo "Number of threads           : "$SLURM_CPUS_PER_TASK
echo "Number of MPI ranks per node: "$SLURM_TASKS_PER_NODE
echo "Number of threads per core  : "$SLURM_THREADS_PER_CORE
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Port                        : "$PORT
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

# Create data:
./bin/release/iRAiSD -n sweepnetd1 -I DATASETS/D1/DATA1_S128_P1000/train/neutral.ms -L 100000 -n data

#sh TrainTest.sh -m SweepNet -c bp -i DATASETS/D1/DATA1_S128_P1000 -o data -h 128 -n 128 -s 5000 -l 100000 -r 5 -e 10 -t black
