#%%
import pandas as pd
from pathlib import Path
#%%
path = Path('../results')

# %%
# Get file with using wildcard
path_full = list(path.glob('Testing_*Full*.csv'))
path_lora = list(path.glob('Testing_*Lora*.csv'))
# %%
df1 = pd.read_csv(path_full[0])
df2 = pd.read_csv(path_full[1])
# %%
merged_df = pd.concat([df2, df1], axis=1)

# %%
merged_df.head()
# %%
merged_df.to_csv(path/'Predictions_Full', index=False)
# %%
df1 = pd.read_csv(path_lora[0])
df2 = pd.read_csv(path_lora[1])
merged_df = pd.concat([df2, df1], axis=1)
merged_df.to_csv(path/'Predictions_Lora', index=False)
# %%
