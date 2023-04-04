"""
In this file we will try to parse the losses of the training process and plot the curve associated to it
"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

def parser():
    """
    Parse the arguments
    Args:
        args:

    Returns:

    """

    parser = argparse.ArgumentParser(description='Parse the logs')
    parser.add_argument('--root', default="/home/ids/benigmim/projects/DAFormer/work_dirs", type=str, help='Path to the log file')
    parser.add_argument('--log_file', type=str, help='Path to the log file')
    return parser.parse_args()


def parse_json_logs(log_file, type="train"):
    """
    Parse the json log file and extract the training losses
    Args:
        log_file: path to the log file
    Returns: list of losses
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
    losses = []
    for line in lines[1:]:
        if "loss" in line and "IoU" in line :
            list_value = re.findall(r'"loss": ([+-]?\d+\.\d*)', line)
            if len(list_value) == 1:
                losses.append(float(list_value[0]))
    plt.plot(losses)
    plt.show()
    return losses


def main():
    args = parser()
    log_path = os.path.join(args.root, args.log_file)
    losses = parse_json_logs(log_path)
    plt.plot(losses)
    plt.show()



if __name__ == '__main__':
    main()