import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

def create_directory(dir):
    """
    Create a directory if it does not exist
    Args:
        dir: the directory to be created

    Returns: None

    """
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created")


def generate_list_ckpt(training_steps,checkpoint_steps):
    """
    Generate a list of checkpoints using the maximum number of training_steps
    Args:
        training_steps: the maximum number of training steps
        checkpoint_steps: the number of steps between each checkpoint

    Returns:

    """
    return [x for x in range(checkpoint_steps, training_steps+1, checkpoint_steps)]