from eval import eval
import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-w", "--weight_path", help="Weight path")
parser.add_argument("-c", "--checkpoint_name", help="Checkpoint name")

# Read arguments from command line
args = parser.parse_args()


## Evaluate
eval(
        weight_path=args.weight_path,
        checkpoint_name=args.checkpoint_name
    )
print("Finished.")

