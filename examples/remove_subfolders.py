# import json
# import os

# # Define the path to the JSON file and the main folder
# json_path = "/home/zubairirshad/Diffusion-SDF/train_sdf/data/splits/couch_all.json"
# main_folder = "/home/zubairirshad/Downloads/shapenet/04256520"


# # Load the JSON file
# with open(json_path) as f:
#     data = json.load(f)

# # Get the list of selected subfolders
# selected_folders = data["acronym"]["Couch"]

# # Iterate over all subfolders in the main folder
# for folder in os.listdir(main_folder):
#     folder_path = os.path.join(main_folder, folder)
    
#     # Check if the subfolder is in the list of selected folders
#     if folder not in selected_folders:
#         # If not, remove the subfolder and all its contents
#         os.system("rm -rf {}".format(folder_path))


#Remove .obj and .mtl files
import json
import os

# Define the path to the JSON file and the main folder
json_path = "/home/ubuntu/zubair/Diffusion-SDF/train_sdf/data/splits/couch_all.json"
main_folder = "/experiments/zubair/shapenet/models"


# Load the JSON file
with open(json_path) as f:
    data = json.load(f)

# Get the list of selected subfolders
selected_folders = data["acronym"]["Couch"]

# Iterate over all subfolders in the main folder
for file in os.listdir(main_folder):
    name = file[0].split('.')[0]
    
    # Check if the subfolder is in the list of selected folders
    if name not in selected_folders:
        # If not, remove the subfolder and all its contents
        file_to_remove = os.path.join(main_folder, file)
        os.system("rm {}".format(file_to_remove))