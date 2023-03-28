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
--continue_from_checkpoint \
--checkpoint_path /data/$USER/checkpoints/esnli-epoch=00-val/loss=1.20 \

