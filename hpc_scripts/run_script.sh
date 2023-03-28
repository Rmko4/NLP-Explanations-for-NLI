#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=ESNLI
#SBATCH --profile=task

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
--model_name google/flan-t5-base \
--data_path $TMPDIR/datasets/esnli \
--checkpoint_path /data/$USER/checkpoints/ \
--fine_tune_mode lora \
--learning_rate 1e-4 \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_epochs 3 \
--log_every_n_steps 200 \
--val_check_interval 1000 \
--limit_val_batches 25 \
--n_text_samples 10 \
--log_every_n_generated 49
