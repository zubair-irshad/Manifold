import os
import shutil
import json

# set the directory path
dir_path = "/experiments/zubair/shapenet/models"

json_path = "/home/ubuntu/zubair/Diffusion-SDF/train_sdf/data/splits/couch_all.json"
# load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# iterate through the subfolders and move the files
for subfolder in data['acronym']['Couch']:
    obj_path = os.path.join(dir_path, subfolder + '.obj')
    mtl_path = os.path.join(dir_path, subfolder + '.mtl')
    if os.path.exists(obj_path):
        shutil.move(obj_path, 'destination_folder')
    if os.path.exists(mtl_path):
        shutil.move(mtl_path, 'destination_folder')