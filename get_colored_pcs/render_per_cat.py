# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import argparse
from tqdm.auto import tqdm

from blender.render import render_model

os.environ["BLENDER_PATH"] = "/root/dev/blender/blender-3.3.1-linux-x64/blender" # your blender path

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', required=True, type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--cat_id', required=True, type=str,
    help='category id to process'
)
parser.add_argument(
    '--dataset_folder', required=True, type=str, default='./tmp',
    help='path for downloaded 3d dataset folder')
parser.add_argument(
    '--num_images', type=int, default=20,
    help='number of rendered images')
parser.add_argument(
    '--camera_pose', type=str, default='random',
    help='camera pose mode'
)
parser.add_argument(
    '--light_mode', type=str, default='random',
    help='light mode'
)
args = parser.parse_args()

# for cat_id in synset_list:
model_ids = sorted(os.listdir(os.path.join(args.dataset_folder, args.cat_id)))
if args.cat_id == "03001627":
    model_ids.remove("901cab3f56c74894d7f7a4c4609b0913") # causes error
if args.cat_id == "02958343":
    model_ids.remove("f9c1d7748c15499c6f2bd1c4e9adb41") # causes error

for idx, model_id in enumerate(tqdm(model_ids)):
    out_dir_path = os.path.join(args.save_folder, args.cat_id, model_id)

    # NOTE: This is for resuming. Check if the existing directories have each zip file!
    if os.path.exists(out_dir_path):
        if len(os.listdir(out_dir_path)) == 1 and os.listdir(out_dir_path)[0].endswith(".zip"):
            print(f"Skipping {idx+1}th instance: {model_id}")
            continue
        elif len(os.listdir(out_dir_path)) == 0:
            out_zip_path = os.path.join(out_dir_path, f"rendered_images.zip")
            render_model(
                model_path = os.path.join(args.dataset_folder, args.cat_id, model_id, "model.obj"),
                output_path = out_zip_path,
                num_images = args.num_images,
                camera_pose = args.camera_pose
            )
            print(f"`{out_dir_path}` processed successfully!\t{idx+1} / {len(model_ids)}")
            continue
        else:
            raise RuntimeError(f"Directory has weird file(s). Check out the directory `{out_dir_path}`.")

    os.makedirs(out_dir_path, exist_ok=True)
    out_zip_path = os.path.join(out_dir_path, f"rendered_images.zip")
    render_model(
        model_path = os.path.join(args.dataset_folder, args.cat_id, model_id, "model.obj"),
        output_path = out_zip_path,
        num_images = args.num_images,
        camera_pose = args.camera_pose
    )
    print(f"`{out_dir_path}` processed successfully!\t{idx+1} / {len(model_ids)}")