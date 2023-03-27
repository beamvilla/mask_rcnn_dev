import os

from train import train
from eval import eval


save_trained_model_dir, checkpoint_name = train()

for f in os.listdir(save_trained_model_dir):
    if f.endswith(".h5"):
        weight_path = os.path.join(save_trained_model_dir, f)
        break

eval(
        weight_path=weight_path,
        checkpoint_name=checkpoint_name
    )

print("Finished.")