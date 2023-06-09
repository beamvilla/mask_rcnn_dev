import os
import shutil

from train import train
from eval import eval


## Train
save_trained_model_dir, checkpoint_name = train()

for f in os.listdir(save_trained_model_dir):
    if f.endswith(".h5"):
        weight_path = os.path.join(save_trained_model_dir, f)
        break

for f in os.listdir("./config"):
    des_dir = os.path.join(save_trained_model_dir, "config")
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    shutil.copy(os.path.join("./config", f), os.path.join(des_dir, f))

## Evaluate
eval(
        weight_path=weight_path,
        checkpoint_name=checkpoint_name
    )
print("Finished.")