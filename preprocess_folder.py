"""
In this file we will delete files which follow a specific pattern in their names
"""
import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a folder of images")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/ids/benigmim/dataset/data_for_daformer",
        help="Path to the input folder",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to the input folder",
    )
    parser.add_argument(
        "--size", type=int, help="Size of the new folder")
    args = parser.parse_args()
    return args


def get_list_files(name_folder):
    mypath = os.path.join(args.root, name_folder)
    return  [f for f in listdir(mypath) if f.endswith(".png")]

def main(args):
    # name_folder = args.input_folder
    list_files = get_list_files(args.input_folder)
    new_list = []
    for file in list_files:
        if int(file.split("_")[1]) < args.size:
            new_list.append(file)
        # new_list.append(file)
    # random.shuffle(new_list)
    # create a new folder
    target_folder = os.path.join(args.root,args.input_folder+f"_{args.size}_samples")
    os.mkdir(target_folder)
    src_folder = os.path.join(args.root,args.input_folder)
    trg_folder = os.path.join(args.root,target_folder)
    for file in new_list[:args.size]:
        shutil.copy(os.path.join(src_folder,file), os.path.join(trg_folder,file))

    # trg_folder = "/home/ids/benigmim/projects/DAFormer/data/asm_stylized_dataset"
    # with open(os.path.join(trg_folder,"train.txt"),"w") as output:
    #     for row in new_list:
    #     # for row in list_files:
    #         output.write(row + "\n")
    print("done")

if __name__ == "__main__":
    args = parse_args()
    main(args)