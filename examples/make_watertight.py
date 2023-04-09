import os
import subprocess
from tqdm import tqdm
import shutil

# Define the commands to run
command1 = "./manifold model.obj temp.watertight.obj -s"
command2 = "./simplify -i temp.watertight.obj -o output.obj -m -r 0.02"

# Traverse the directory tree and find all the model.obj files
model_paths = []
for root, dirs, files in os.walk('/home/zubairirshad/Downloads/shapenet/04256520'):
    for file in files:
        if file.endswith('model.obj'):
            # Construct the full path to the model.obj file
            model_path = os.path.join(root, file)
            model_paths.append(model_path)

# Run the commands on all the model.obj files sequentially
for model_path in tqdm(model_paths):
    os.chdir('/home/zubairirshad/Manifold/build')
    # Construct the commands to run on this model.obj file
    print("model_path", model_path)
    if os.path.exists('temp.watertight.obj'):
        os.remove('temp.watertight.obj')
    cmd1 = command1.replace('model.obj', model_path)
    cmd2 = command2.replace('output.obj', os.path.join(os.path.dirname(model_path), 'output.obj'))

    print("cmd1", cmd1)
    print("cmd2", cmd2)
    # Run the commands in parallel
    p1 = subprocess.Popen(cmd1, shell=True)
    p2 = subprocess.Popen(cmd2, shell=True)
    p1.wait()
    p2.wait()

    # output_path = os.path.join(os.path.dirname(model_path), 'output.obj')
    # if os.path.exists(output_path):
    #     shutil.move(output_path, os.path.dirname(model_path))