from parse_args_T5_run import get_args
from evaluate_t5 import test

# Make sure to login to wandb before running this script
# Run: wandb login

# Will call evaluate function from evaluate_t5.py
# but exports predictions instead of score.
if __name__ == "__main__":
    hparams = get_args()
    hparams.predict = True
    test(hparams)
