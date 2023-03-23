from datasets import load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer


class ESNLIDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        dataset_path: str = None,
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

        self.dataset_name_or_path = 'esnli' \
            if dataset_path is None else dataset_path

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        # Downloads and processes the dataset if not on disk
        self._load_processed_dataset()

    def setup(self, stage: str = None) -> None:
        # Should load the now cached or manually saved to disk dataset
        self.datasets = self._load_processed_dataset()

        # Sets the data collator for creating batches
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, padding=True, max_length=self.max_source_length, label_pad_token_id=-100)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            shuffle=True,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['validation'],
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
        )

    def _load_processed_dataset(self):
        # Note this function does not change state of self
        # Downloads the dataset if not on disk

        def load_and_process():
            datasets = load_dataset('esnli')

            raw_dataset_columns = datasets['train'].column_names

            for split in datasets.keys():
                datasets[split] = datasets[split].map(
                    lambda examples: self._preprocess_function(
                        examples, training=(split == "train")),
                    batched=True,
                    remove_columns=raw_dataset_columns
                )

            datasets.set_format(type='torch')
            return datasets

        if self.dataset_name_or_path == 'esnli':
            datasets = load_and_process()
        else:
            try:
                datasets = load_from_disk(self.dataset_name_or_path)
            except FileNotFoundError:
                datasets = load_and_process()
                datasets.save_to_disk(self.dataset_name_or_path)

        return datasets

    def _preprocess_function(self, examples, training=True):
        # Create input text by combining premise and hypothesis
        input_text = [
            f"premise: {premise}\n" f"hypothesis: {hypothesis}"
            for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
        ]

        # Tokenize input text
        model_inputs = self.tokenizer(
            input_text, truncation=True, max_length=self.max_source_length)

        # Tokenize first explanation and add as "labels" to model inputs
        targets = self.tokenizer(
            examples['explanation_1'], truncation=True, max_length=self.max_target_length)

        model_inputs["labels"] = targets["input_ids"]

        # Tokenize all explanations and assign to explanation_i
        if not training:
            for i in range(1, 4):
                key_explanation = f"explanation_{i}"
                targets = self.tokenizer(
                    examples[key_explanation], truncation=True, padding='max_length', max_length=self.max_target_length)
                model_inputs[key_explanation] = targets["input_ids"]
                # Note that these are zero padded and not -100 padded

        return model_inputs


if __name__ == "__main__":
    dm = ESNLIDataModule()
    dm.prepare_data()
    dm.setup()

    # Prints the first batch of the training set
    print(next(iter(dm.train_dataloader())))
