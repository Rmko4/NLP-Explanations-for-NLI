import nltk
import numpy as np
import datasets
from transformers import T5Tokenizer


def preprocess_function(examples, prefix, tokenizer):
    # inputs = [prefix + ' \n ' + 'premise: ' + doc for doc in examples['premise']]
    # inputs2 = [doc for doc in examples['hypothesis']]

    inputs = prefix + ' \n ' + 'premise: ' + examples['premise'] + ' \n ' + 'hypotheses: ' + examples['hypothesis']

    print(inputs)
    # model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    #     labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs
    pass


def compute_metrics(eval_pred):
    # predictions, labels = eval_pred
    # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # # Rouge expects a newline after each sentence
    # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # # Extract a few results
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # # Add mean generated length
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    # result["gen_len"] = np.mean(prediction_lens)

    # return {k: round(v, 4) for k, v in result.items()}
    pass


if __name__ == '__main__':
    raw_datasets = datasets.load_dataset("esnli")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # test = raw_datasets['train'][:5].map(preprocess_function)
    # preprocess_function(raw_datasets['train'][:1], 'test', tokenizer)
