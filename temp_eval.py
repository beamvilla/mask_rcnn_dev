import os
import argparse

from eval import eval
 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-w", "--weight_path", help="Weight path")
parser.add_argument("-c", "--checkpoint_name", help="Checkpoint name")

# Read arguments from command line
args = parser.parse_args()

weight_dir = "/".join(args.weight_path.split("/")[:-1])
eval_config_path = os.path.join(weight_dir, "config", "eval.json")
mrcnn_config_path = os.path.join(weight_dir, "config", "mrcnn_config.json")

## Evaluate
eval(
        weight_path=args.weight_path,
        checkpoint_name=args.checkpoint_name,
        eval_config_path=eval_config_path,
        mrcnn_config_path=mrcnn_config_path
    )
print("Finished.")

