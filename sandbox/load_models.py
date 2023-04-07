#%%
from t5_lit_module import LitT5
from t5_lit_classify import LitT5Classify
import torch
import os

# %%
checkpoint_path_classifier = '~/models/checkpoints/model_lora_classifier.ckpt'
checkpoint_path_main_model = '~/models/checkpoints/model_lora.ckpt'

checkpoint_path_classifier = os.path.expanduser(checkpoint_path_classifier)
checkpoint_path_main_model = os.path.expanduser(checkpoint_path_main_model)


# %%
checkpoint = torch.load(checkpoint_path_classifier, map_location=torch.device('cpu'))

# %%
hyper_params = checkpoint['hyper_parameters']
hyper_params['checkpoint_path_main_model'] = hyper_params.pop('checkpoint_path')


# %%
hyper_params
# %%
torch.save(checkpoint, checkpoint_path_classifier)

# %%



model = LitT5Classify.load_from_checkpoint(
    checkpoint_path=checkpoint_path_classifier,
    checkpoint_path_main_model=checkpoint_path_main_model,

)
# %%
