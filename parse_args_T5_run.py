import argparse
from typing import Union


def get_args():
    parser = argparse.ArgumentParser(
        description='T5 ESNLI Fine-Tuning and Testing')

    # Add arguments here
    parser.add_argument('--model_name', type=str, default='google/flan-t5-base',
                        help='The name or path of the pre-trained T5 model to fine-tune.')
    parser.add_argument('--data_path', type=str, default='~/datasets/esnli/',
                        help='The path to the ESNLI dataset.')
    parser.add_argument('--run_name', type=str, default='Fine-Tuning',
                        help='The name of the run.')
    parser.add_argument('--checkpoint_load_path', type=str, default=None,
                        help='The path to the directory where checkpoint will be loaded from.')
    parser.add_argument('--checkpoint_save_path', type=str, default='~/models/',
                        help='The path to the directory where checkpoints will be saved.')
    parser.add_argument('--results_save_path', type=str, default='results/',
                        help='The path to the directory where results will be saved.')
    parser.add_argument('--fine_tune_mode', type=str, default='full',
                        help='The mode to use for fine-tuning. Can be one of "full", "lora", or "gradual_unfreezing".')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='The learning rate to use for training.')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='The batch size to use for training.')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='The batch size to use for evaluation.')
    parser.add_argument('--max_epochs', type=int, default=3,
                        help='The maximum number of epochs to train for.')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                        help='The number of training steps to log after.')
    parser.add_argument('--val_check_interval', type=int, default=1000,
                        help='The number of training steps between each validation run.')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='The number of batches to use for validation.')
    parser.add_argument('--limit_test_batches', type=int, default=None,
                        help='The number of batches to use for testing.')
    parser.add_argument('--limit_predict_batches', type=int, default=None,
                        help='The number of batches to use for prediction.')
    parser.add_argument('--n_text_samples', type=int, default=10,
                        help='The number of samples to generate for each logging interval.')
    parser.add_argument('--log_every_n_generated', type=int, default=20,
                        help='The number of training steps between each logging of generated samples.')
    parser.add_argument('--lora_r', type=int, default=8, help='LORA R value')
    parser.add_argument('--lora_alpha', type=int,
                        default=32, help='LORA alpha value')
    parser.add_argument('--lora_dropout', type=float,
                        default=0.1, help='LORA dropout value')
    parser.add_argument('--classify', type=bool, default=False,
                        help='Wether to load the dataset with int labels for classification')
    
    args = parser.parse_args()
    return args
