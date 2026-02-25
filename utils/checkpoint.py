import json
import os


def load_checkpoint(results_dir):
    """Return the index to resume from (0 if no checkpoint exists)."""
    checkpoint_path = os.path.join(results_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        start_idx = data.get("completed", 0)
        print(f"Resuming from image {start_idx}")
        return start_idx
    return 0


def save_checkpoint(results_dir, completed, last_prompt=""):
    """Write checkpoint.json with the number of completed images."""
    checkpoint_path = os.path.join(results_dir, "checkpoint.json")
    with open(checkpoint_path, "w") as f:
        json.dump({"completed": completed, "last_prompt": last_prompt}, f)
