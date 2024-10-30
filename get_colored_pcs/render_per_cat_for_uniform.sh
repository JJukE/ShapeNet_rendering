# ["chair": 03001627, "airplain": 02691156, "car": 02958343]
python render_per_cat.py \
    --save_folder /root/hdd2/rendered_shapenet_uniform_light \
    --dataset_folder /root/hdd2/ShapeNetCore.v1 \
    --num_images 24 \
    --light_mode "uniform" \
    --cat_id "02958343"