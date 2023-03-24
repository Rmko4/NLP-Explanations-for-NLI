import argparse


def get_args():
    parser = argparse.ArgumentParser(description='T5 ESNLI Fine-Tuning')

    # Add arguments here
    parser.add_argument('--model_name', type=str, default='google/flan-t5-base',
                        help='The name or path of the pre-trained T5 model to fine-tune.')
    parser.add_argument('--data_path', type=str, default='~/datasets/esnli/',
                        help='The path to the ESNLI dataset.')
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
    parser.add_argument('--limit_val_batches', type=int, default=50,
                        help='The number of batches to use for validation.')
    parser.add_argument('--n_text_samples', type=int, default=10,
                        help='The number of samples to generate for each logging interval.')
    parser.add_argument('--log_every_n_generated', type=int, default=20,
                        help='The number of training steps between each logging of generated samples.')

    args = parser.parse_args()
    return args
