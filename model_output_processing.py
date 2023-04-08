import pandas as pd

def export_predictions(out, hparams, run_name):
    if hparams.classify:
        # Remove batch dimension
        reference_labels = [batch['reference_label'] for batch in out]
        predicted_labels = [batch['predicted_label'] for batch in out]

        # Flatten lists
        reference_labels = [
            item for sublist in reference_labels for item in sublist]
        predicted_labels = [
            item for sublist in predicted_labels for item in sublist]

        # Make pandas dataframe out of reference and predicted label
        df = pd.DataFrame(
            list(zip(reference_labels, predicted_labels)),
            columns=['reference_label', 'predicted_label']
        )

        # Save as csv
        df.to_csv(
            f"{hparams.results_save_path}/{run_name}.csv", index=False)
    else:
        # Remove batch dimension
        input_texts = [batch['input_text'] for batch in out]
        generated_texts = [batch['generated_text'] for batch in out]
        reference_texts = [batch['reference_texts'] for batch in out]

        # Flatten lists
        input_texts = [item for sublist in input_texts for item in sublist]
        generated_texts = [
            item for sublist in generated_texts for item in sublist]

        # Flatten reference texts
        flattened_reference_texts = []
        for batch in reference_texts:
            reference_texts_t = list(map(list, zip(*batch)))
            flattened_reference_texts.extend(reference_texts_t)

        reference_texts = list(map(list, zip(*flattened_reference_texts)))

        # Make pandas dataframe out of input, generated and reference texts
        df = pd.DataFrame(
            list(zip(input_texts, generated_texts,
                    reference_texts[0], reference_texts[1], reference_texts[2])),
            columns=['input_text', 'generated_text', 'reference_texts_0',
                        'reference_texts_1', 'reference_texts_2']
        )

        # Save as csv
        df.to_csv(
            f"{hparams.results_save_path}/{run_name}.csv", index=False)
