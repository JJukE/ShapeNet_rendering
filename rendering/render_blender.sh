find /root/hdd2/ShapeNetCoreV2/no_mtl_objs/ -name *model_normalized_nomtl.obj \
-exec ../blender-2.90.0-linux64/blender --background --python ./render_blender.py -- \
--output_folder /root/hdd2/ShapeNetCoreV2/render_for_gen_desc {} \;