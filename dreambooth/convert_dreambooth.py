# ---------------------------------------------------------------
# Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


"""
This file converts a checkpoint from the training script into a checkpoint that can be used for inference.
"""

import os
from accelerate import Accelerator
from diffusers import DiffusionPipeline
import argparse
from my_utils import create_directory, generate_list_ckpt

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--filepath",
        type=str,
        help="path of the model",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    args, _ = parser.parse_known_args()
    return args

def main(args):

    # Load the pipeline with the same arguments (model, revision) that were used for training
    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    accelerator = Accelerator()

    # Use text_encoder if `--train_text_encoder` was used for the initial training
    unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)
    # unet = accelerator.prepare(pipeline.unet)

    # Restore state from a checkpoint path. You have to use the absolute path here.
    training_path = os.path.join("logs/checkpoints", args.filepath)
    liste_steps = generate_list_ckpt(args.max_train_steps,args.checkpointing_steps)
    for steps in liste_steps:

        checkpoint_path = os.path.join(training_path, f"checkpoint-{steps}")
        accelerator.load_state(checkpoint_path)

        # Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )

        inference_checkpoint_path = os.path.join(training_path, f"inf_ckpt{steps}")

        # Perform inference, or save, or push to the hub
        pipeline.save_pretrained(inference_checkpoint_path)
        print("Saved to", inference_checkpoint_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)