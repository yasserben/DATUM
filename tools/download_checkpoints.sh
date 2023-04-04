# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

# Instructions for Manual Download:
#
# Please, download the [MiT weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)
# pretrained on ImageNet-1K provided by the official
# [SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a
# folder `pretrained/` within this project. For most of the experiments, only
# mit_b5.pth is necessary.

# Automatic Downloads:
set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
gdown --id 1d3wU8KNjPL4EqMCIEO_rO-O3-REpG82T  # MiT-B3 weights
gdown --id 1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2  # MiT-B4 weights
gdown --id 1d7I50jVjtCddnhpf-lqj8-f13UyCzoW1  # MiT-B5 weights
cd ../

