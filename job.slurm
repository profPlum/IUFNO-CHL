#!/bin/bash
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --nodes=1                            # Trying allocating a whole node to play nice with pytorch-lightning
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --mem-per-gpu=115GB                  # CPU? RAM per GPU < 1/4 of the total node RAM = 120GB
#SBATCH --gpus-per-node=1                    # GPUs per task is 1
#SBATCH --cpus-per-task=72                   # CORES per task for data loaders (72 avail per GPU) (very few really used)
#SBATCH --time=04:00:00                      # Maximum runtime (HH:MM:SS) (48 hours for WNO-600-epochs)
#SBATCH --partition=ghx4                     # Partition name
#SBATCH --account=beoi-dtai-gh
#SBATCH --job-name=1_flow_thru_IUFNO_long_test

# Not going to work anymore b/c we would need to save the entire checkpoint which causes OOM...
# ##SBATCH --signal=SIGUSR1@90
# ##SBATCH --time=1-00:00:00                    # Maximum runtime (HH:MM:SS)

# Load modules or environment
. ~/.bashrc
conda activate uqops
#module load cuda/11.8.0

# Optional: Debugging info
echo "Job started on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "batch node ip: $(host $(hostname))"
echo "rank hostnames & Ips:"
srun bash -c 'echo host: $SLURMD_NODENAME, rank: $PMI_RANK, slurm_proc_id: $SLURM_PROCID, $(host $(hostname))'
echo srun nvidia-smi:
srun --ntasks-per-node=1 nvidia-smi

# Run your PyTorch Lightning script
srun --kill-on-bad-exit -n1 python IUFNO.py

# Optional: Debugging info
echo "Job finished at $(date)"
echo done with job time remaining:
squeue -h -j $SLURM_JOB_ID -O TimeLeft
