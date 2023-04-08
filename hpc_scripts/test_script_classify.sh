#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=test
#SBATCH --mem=64GB
#SBATCH --profile=task

# Use scratch due to limited space on /home
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb
export TOKENIZERS_PARALLELISM=true

# Copy git repo to local
cp -r ~/NLP-Explanations-for-NLI/ $TMPDIR

module load Python
module load cuDNN
module load CUDA
source /scratch/$USER/envs/nlpenv/bin/activate

# cd to working directory (repo)
cd $TMPDIR/NLP-Explanations-for-NLI/

python3 evaluate_t5.py \
--model_name google/flan-t5-base \
--data_path /scratch/$USER/datasets/esnli_classify \
--checkpoint_load_path /scratch/$USER/checkpoints/model_lora_classifier \
--results_save_path /scratch/$USER/results/ \
--eval_batch_size 32 \
--log_every_n_steps 200 \
--run_name Testing_Classify_Lora \
--classify True
