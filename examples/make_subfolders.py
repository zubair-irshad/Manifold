import os

folder_path = "/experiments/zubair/shapenet/couch"  # replace with the path to your folder
for filename in os.listdir(folder_path):
    if filename.endswith(".obj"):
        obj_path = os.path.join(folder_path, filename)
        model_folder = filename[:-4]
        model_path = os.path.join(folder_path, model_folder, "model.obj")
        os.makedirs(os.path.join(folder_path, model_folder), exist_ok=True)
        os.rename(obj_path, model_path)
    elif filename.endswith(".mtl"):
        mtl_path = os.path.join(folder_path, filename)
        model_folder = filename[:-4]
        model_path = os.path.join(folder_path, model_folder, "model.mtl")
        os.makedirs(os.path.join(folder_path, model_folder), exist_ok=True)
        os.rename(mtl_path, model_path)