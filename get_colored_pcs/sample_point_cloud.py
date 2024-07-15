# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import argparse

import blobfile as bf
from blender.render import render_model
from view_data import BlenderViewData
from point_cloud import PointCloud
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', required=True, type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--rendered_folder', required=True, type=str, default='./tmp',
    help='path for processed rendered images folder')
parser.add_argument(
    '--num_images', type=int, default=20,
    help='number of rendered images')
parser.add_argument(
    '--num_pts', type=int, default=20480,
    help='number of points to sample from RGBAD images'
)
args = parser.parse_args()

# TODO: modify
cat_list = [
    # '02691156', # Plane -> added
    # '02958343', # Car
    '03001627', # Chair
    # '04379243', # Table -> added
    # '03790512' # Motorbike
]

for cat_id in cat_list:
    model_ids = sorted(os.listdir(os.path.join(args.rendered_folder, cat_id)))

    if cat_id == "03001627":
        model_ids.remove("901cab3f56c74894d7f7a4c4609b0913") # causes error
    if cat_id == "02958343":
        model_ids.remove("f9c1d7748c15499c6f2bd1c4e9adb41") # causes error

    for idx, model_id in enumerate(tqdm(model_ids)):
        out_dir_path = os.path.join(args.save_folder, cat_id, model_id)
        if os.path.exists(out_dir_path) and len(os.listdir(out_dir_path)) > 0:
            if os.listdir(out_dir_path)[0].endswith(".npz") or os.listdir(out_dir_path)[0].endswith(".ply"):
                print(f"{out_dir_path} has already processed!")
                continue
        os.makedirs(out_dir_path, exist_ok=True)

        rendered_zip_path = os.path.join(args.rendered_folder, cat_id, model_id, f"rendered_images.zip")

        vd = BlenderViewData(rendered_zip_path)
        pc = PointCloud.from_rgbd(vd, args.num_images)
        
        # sampled_pc = pc.farthest_point_sample(args.num_pts) # too long time
        sampled_pc = pc.random_sample(args.num_pts)
        
        # with bf.BlobFile(os.path.join(out_dir_path, f"colored_pc_{args.num_pts}.ply"), "wb") as writer:
        #     sampled_pc.write_ply(writer)
        
        sampled_pc.save(os.path.join(out_dir_path, f"colored_pc_{args.num_pts}.npz"))
        
