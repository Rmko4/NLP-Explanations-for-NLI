#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=FULL_ESNLI
#SBATCH --profile=task

# Use scratch due to limited space on /home
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb

# Copy git repo to local
cp -r ~/NLP-Explanations-for-NLI/ $TMPDIR

module load Python
module load cuDNN
module load CUDA
source /scratch/$USER/.envs/nlpenv/bin/activate

# cd do working directory (repo)
cd $TMPDIR/NLP-Explanations-for-NLI/

python3 train_t5.py \
--model_name google/flan-t5-base \
--data_path /scratch/$USER/datasets/esnli_classify \
--checkpoint_save_path /scratch/$USER/checkpoints/ \
--classify True \
--fine_tune_mode full \
--learning_rate 1e-4 \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_epochs 10 \
--log_every_n_steps 200 \
--val_check_interval 1000 \

