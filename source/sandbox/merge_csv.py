#%%
import pandas as pd
from pathlib import Path
#%%
path = Path('../results')

# %%
# Get files using wildcard
path_full = list(path.glob('Testing_*Full*.csv'))
path_lora = list(path.glob('Testing_*Lora*.csv'))
# %%
df1 = pd.read_csv(path_full[0])
df2 = pd.read_csv(path_full[1])
# %%
# Swap the two columns in df1
df1 = df1[['predicted_label', 'reference_label']]

# %%
import datasets
# %%
int2str_label = datasets.load_dataset("esnli")['test'].features['label'].int2str
# %%
# Map the int2str_label function to 'reference_label and 'predicted_label' columns
df1['reference_label'] = df1['reference_label'].map(int2str_label)
df1['predicted_label'] = df1['predicted_label'].map(int2str_label)
# %%
merged_df = pd.concat([df2, df1], axis=1)

# %%
merged_df.head()
# %%
merged_df.to_csv(path/'Predictions_Full.csv', index=False)
# %%
df1 = pd.read_csv(path_lora[0])
df2 = pd.read_csv(path_lora[1])
df1 = df1[['predicted_label', 'reference_label']]
df1['reference_label'] = df1['reference_label'].map(int2str_label)
df1['predicted_label'] = df1['predicted_label'].map(int2str_label)
merged_df = pd.concat([df2, df1], axis=1)
merged_df.to_csv(path/'Predictions_Lora.csv', index=False)
# %%
merged_df.head()
# %%
pass