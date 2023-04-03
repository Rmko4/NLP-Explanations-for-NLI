#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=Classify-Train
#SBATCH --profile=task

# Use scratch due to limited space on /home
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb

# Copy git repo to local
cp -r ~/NLP-Explanations-for-NLI/ $TMPDIR

module load Python
module load cuDNN
module load CUDA
source /scratch/$USER/envs/nlpenv/bin/activate

# cd to working directory (repo)
cd $TMPDIR/NLP-Explanations-for-NLI/

python3 train_t5.py \
--model_name google/flan-t5-base \
--data_path /scratch/$USER/datasets/esnli_classify \
--run_name Classify-Train \
--checkpoint_save_path /scratch/$USER/checkpoints/ \
--checkpoint_load_path /scratch/$USER/checkpoints/model_full \
--classify True \
--learning_rate 1e-4 \
--train_batch_size 16 \
--eval_batch_size 16 \
--max_epochs 10 \
--log_every_n_steps 200 \
--val_check_interval 1000 \
