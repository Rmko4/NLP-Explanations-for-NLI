# Evaluating Explanations in Natural Language Inference
Fine-tuning and evaluation of a language model for explanation generation of natural language inference. Fine-tuning scripts for a pre-trained T5 model supporting both full model fine-tuning as well as [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA) fine-tuning using [🤗 PEFT](https://github.com/huggingface/peft) are included in this repository. Furthermore, a testing script for evaluating and generating predictions is included. A probing classifier may additionally be trained and evaluated for getting insights into the correspondence of the inference label with the generated explanations.

## Overview
### Data
The dataset that is used for fine-tuning and evaluating the models is the [e-SNLI](https://huggingface.co/datasets/esnli) dataset. This dataset extends the Standford Natural Language Inference dataset by including explanations for the entailment relations.

The following features are present for each data point:
- Premise: a string feature.
- Hypothesis: a string feature.
- Label: a classification label, with possible values including: 
  - *entailment* (0),
  - *neutral* (1),
  - *contradiction* (2).
- Explanation_1: a string feature.
- Explanation_2: a string feature.
- Explanation_3: a string feature.

The code for the data and pre-processing is encapsulated in [ESNLIDataModule](esnli_data.py). This class extends [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html?highlight%3Ddatamodule) of the PyTorch Lightning framework.

### Model
The model that is trained for the task of explanation generation for natural language inference (NLI) is the [T5 model](https://huggingface.co/docs/transformers/model_doc/t5). The T5 model is an encoder-decoder model. The Flan-T5-Base model has been extensively used for the project. However, any pre-trained T5 model is supported. A diagram of the complete architecture is shown below.

![Model architecture diagram](images/model_architecture.png)
![T5 Block](images/T5_block.png)

The main model with the Language Modelling head (LM Head) is wrapped in used to generate the explanations. The model with the 

## Running the code
### Installation
To install all dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### Pre-processing data
The code for training the model performs data preprocessing, so there is no need to preprocess the data separately.

### Training the Models
To train the model, run the following command:

css
Copy code
python train.py [options]
Possible options include:

--model_type: the type of model to use (default: esnli)
--num_epochs: the number of epochs to train for (default: 10)
--batch_size: the batch size to use during training (default: 32)
--learning_rate: the learning rate to use during training (default: 0.001)
The code will write the results to a WANDB dashboard and to standard out.

### Using Pre-trained Models for Prediction
To use a pre-trained model for prediction, run the following command:

css
Copy code
python predict.py [options]
Possible options include:

--model_type: the type of pre-trained model to use (default: esnli)
--model_path: the path to the pre-trained model file (default: ./models/esnli_model.pt)
--data_path: the path to the data file to predict (default: ./data/test.csv)
--output_path: the path to the output file (default: ./results/predictions.csv)
The code will write the predictions to a CSV file.

### Evaluation
The code provides metrics for the model and cannot be customized.
