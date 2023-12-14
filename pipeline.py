import os
import shutil

from train import train
from eval import eval
from custom_config import dir_path


## Train
save_trained_model_dir, checkpoint_name = train()

for f in os.listdir(save_trained_model_dir):
    if f.endswith(".h5"):
        weight_path = os.path.join(save_trained_model_dir, f)
        break

config_dir = os.path.join(dir_path, "config")

for f in os.listdir(config_dir):
    des_dir = os.path.join(save_trained_model_dir, "config")
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    shutil.copy(os.path.join(config_dir, f), os.path.join(des_dir, f))


weight_dir = "/".join(weight_path.split("/")[:-1])
eval_config_path = os.path.join(weight_dir, "config", "eval.json")
mrcnn_config_path = os.path.join(weight_dir, "config", "mrcnn_config.json")

## Evaluate
eval(
        weight_path=weight_path,
        checkpoint_name=checkpoint_name,
        eval_config_path=eval_config_path,
        mrcnn_config_path=mrcnn_config_path
    )
print("Finished.")