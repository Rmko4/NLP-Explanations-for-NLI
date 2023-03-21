from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer


class ESNLIDataModule(LightningDataModule):

    def __init__(self,
                 model_name_or_path: str = "google/flan-t5-small",
                 train_batch_size: int = 16,
                 eval_batch_size: int = 64,
                 max_source_length: int = 512,
                 max_target_length: int = 128,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        # Downloads the dataset if not on disk
        load_dataset("esnli")
        T5Tokenizer.from_pretrained(self.model_name_or_path)

    def setup(self, stage: str = None):
        # Loads the dataset and tokenizer
        self.datasets = load_dataset("esnli")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)

        raw_dataset_columns = self.datasets['train'].column_names

        # Applies _preprocess_function to all splits (taking care of the training argument)
        # After all, train dataset only has one explanation.
        for split in self.datasets.keys():
            self.datasets[split] = self.datasets[split].map(
                lambda examples: self._preprocess_function(
                    examples, training=(split == "train")),
                batched=True,
                remove_columns=raw_dataset_columns
            )

        self.datasets.set_format(type='torch')

        # Sets the data collator for creating batches
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, padding=True, label_pad_token_id=-100)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], shuffle=True, batch_size=self.train_batch_size, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.datasets['validation'], batch_size=self.eval_batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.eval_batch_size, collate_fn=self.data_collator)

    def _preprocess_function(self, examples, training=True):
        input_text = ['premise: ' + premise + ' \n ' + 'hypothesis: ' + hypothesis
                      for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])]

        model_inputs = self.tokenizer(
            input_text, truncation=True, max_length=self.max_source_length)

        if training:
            target_text = examples['explanation_1']
            targets = self.tokenizer(
                target_text, truncation=True, max_length=self.max_target_length)

            model_inputs["labels"] = targets["input_ids"]
        else:
            for i in range(1, 4):
                key_explanation = 'explanation_' + str(i)
                target_text = examples[key_explanation]
                targets = self.tokenizer(
                    target_text, truncation=True, max_length=self.max_target_length)
                model_inputs[key_explanation] = targets["input_ids"]

        return model_inputs


if __name__ == "__main__":
    dm = ESNLIDataModule()
    dm.prepare_data()
    dm.setup()

    # Prints the first batch of the training set
    print(next(iter(dm.train_dataloader())))