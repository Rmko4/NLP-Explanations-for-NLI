# Use scratch due to limited space on /home
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb

python3 test_t5.py \
--model_name google/flan-t5-base \
--data_path $TMPDIR/datasets/esnli \
--checkpoint_save_path /data/$USER/checkpoints/esnli-epoch=00-val/loss=1.20.ckpt \
--eval_batch_size 32 \
--log_every_n_steps 200 \
--limit_test_batches 25 \
--limit_predict_batches 25