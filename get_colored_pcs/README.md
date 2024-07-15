Getting colored point clouds by rendering RGBD images with blender and post-processing it.

Codes are from [Shape-E](https://github.com/openai/shap-e/tree/main/shap_e/rendering)

# Render 3D Models as RGBD Images

- Download shapenet V1 dataset following the [official link](https://shapenet.org/) and
  unzip the downloaded file `unzip SHAPENET_SYNSET_ID.zip`.
- Download Blender following the [official link](https://www.blender.org/), we used
  Blender **v3.3.1**, we haven't tested on other versions.

- Set environment variable:
```bash
export BLENDER_PATH=PATH_TO_BLENDER
```
  - for example, ```export BLENDER_PATH=/root/dev/blender/blender-3.3.1-linux-x64/blender```

- Running the render script:

```bash
python render_all.py --save_folder PATH_TO_SAVE --dataset_folder PATH_TO_3D_OBJ --num_images NUM_IMAGES
```

- The number of image(20) is from [point-e](https://arxiv.org/abs/2212.08751)

# Post-process to get Point Clouds

Sample colored point cloud.

Original code gets the point cloud with farthest point sampling, but it takes too much time.

So I use random sampling, instead.