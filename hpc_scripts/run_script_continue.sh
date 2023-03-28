#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=ESNLI
#SBATCH --profile=task

# Use scratch due to limited space on /home
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb

# Copy git repo to local
cp -r ~/NLP-Explanations-for-NLI/ $TMPDIR

mkdir -p $TMPDIR/datasets/esnli

cp -r /data/$USER/datasets/esnli $TMPDIR/datasets/esnli

module load Python
module load cuDNN
module load CUDA
source /data/$USER/.envs/nlpenv/bin/activate

# cd do working directory (repo)
cd $TMPDIR/NLP-Explanations-for-NLI/

python3 train_t5.py \
--checkpoint_load_path /data/$USER/checkpoints/esnli-epoch=00-val/loss=1.20.ckpt
--model_name google/flan-t5-base \
--data_path /data/$USER/datasets/esnli \
--checkpoint_save_path /data/$USER/checkpoints/ \
--fine_tune_mode full \
--learning_rate 1e-4 \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_epochs 10 \
--log_every_n_steps 200 \
--val_check_interval 1000 \
--limit_val_batches 25 \
--n_text_samples 10 \
--log_every_n_generated 49

