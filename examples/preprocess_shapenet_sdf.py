import os
# Enable this when running on a computer without a screen:
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import traceback

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ':0'
import sys
# sys.path.append('/home/zubairirshad/Manifold/mesh_to_sdf')
sys.path.append('/home/ubuntu/zubair/Manifold/mesh_to_sdf')
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, scale_to_unit_sphere, BadMeshException, sample_sdf_near_surface
import random
from utils import *
from pathlib import Path
from os import path
# import open3d as o3d
# DATAFILES_FOLDER = '/home/zubairirshad/Downloads/couch'
# SAVE_FOLDER = '/home/zubairirshad/Downloads/couch_sdf'

DATAFILES_FOLDER = '/experiments/zubair/shapenet/couch'
SAVE_FOLDER = '/experiments/zubair/shapenet/couch_sdf'
SDF_POINT_CLOUD_SIZE = 235000 # For DeepSDF point clouds (CREATE_SDF_CLOUDS)
# POINT_CLOUD_SAMPLE_SIZE = 64**3 # For uniform and surface points (CREATE_UNIFORM_AND_SURFACE)

# Options for virtual scans used to generate SDFs
USE_DEPTH_BUFFER = True
SCAN_COUNT = 50
SCAN_RESOLUTION = 500


def compute_trimesh_centroid(obj):
  bounds = obj.bounds
  bounds_centroid = np.array([(bounds[1][0] - bounds[0][0]) / 2.0 + bounds[0][0],
                              (bounds[1][1] - bounds[0][1]) / 2.0 + bounds[0][1],
                              (bounds[1][2] - bounds[0][2]) / 2.0 + bounds[0][2]])
  return bounds_centroid

def sample_shapenet_object(obj_path):
  obj_trimesh = trimesh.load(obj_path)
  # Translated object to center.
  center_to_origin = np.eye(4)
  center_to_origin[0:3, 3] = -compute_trimesh_centroid(obj_trimesh)
  obj_trimesh.apply_transform(center_to_origin)
  # Set largest dimension so that object is contained in a unit cube.
  bounding_box = obj_trimesh.bounds
  
  current_scale = np.array([
      bounding_box[1][0] - bounding_box[0][0], bounding_box[1][1] - bounding_box[0][1],
      bounding_box[1][2] - bounding_box[0][2]
  ])
  scale_matrix = np.eye(4)
  scale_matrix[0:3, 0:3] = scale_matrix[0:3, 0:3] * 1.0 / np.max(current_scale)
  obj_trimesh.apply_transform(scale_matrix)  
  return obj_trimesh

def get_uniform_and_surface_points(surface_point_cloud, number_of_points = 231000):
        unit_sphere_points = np.random.uniform(-1, 1, size=(number_of_points * 2, 3)).astype(np.float32)
        unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
        uniform_points = unit_sphere_points[:number_of_points, :]

        distances, indices = surface_point_cloud.kd_tree.query(uniform_points)
        uniform_sdf = distances.astype(np.float32).reshape(-1) * -1
        uniform_sdf[surface_point_cloud.is_outside(uniform_points)] *= -1

        surface_points = surface_point_cloud.points[indices[:, 0], :]
        near_surface_points = surface_points + np.random.normal(scale=0.0025, size=surface_points.shape).astype(np.float32)
        near_surface_sdf = surface_point_cloud.get_sdf(near_surface_points, use_depth_buffer=USE_DEPTH_BUFFER)
        
        model_size = np.count_nonzero(uniform_sdf < 0) / number_of_points
        if model_size < 0.01:
            raise BadMeshException()

        return uniform_points, uniform_sdf, near_surface_points, near_surface_sdf

def process_model_file(filename, DIRECTORY_SDF_CLOUD):
    try:
        mesh = sample_shapenet_object(filename)
        try:
            surface_point_cloud = get_surface_point_cloud(mesh, bounding_radius=1, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
            pc = surface_point_cloud.points

            # print("pc", pc.shape)
            pc_size = 1024
            pc_idx = np.random.permutation(pc.shape[0])[:pc_size]
            pc = pc[pc_idx]

            # print("pc", pc.shape)

            # # o3d.visualization.draw_geometries([line_set], zoom=0.8)
            # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc)), line_set])

            # sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)

            sdf_points_surface, sdf_values_surface = surface_point_cloud.sample_sdf_surface(sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)
            sdf_points_uniform, sdf_values_uniform = surface_point_cloud.sample_sdf_uniform(sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)
            
            # print("sdf_points_surface", sdf_points_surface.shape, sdf_values_surface.shape)
            # print("sdf_points_uniform", sdf_points_uniform.shape, sdf_values_uniform.shape)
            # print("sdf_points", sdf_points.shape, sdf_values.shape)
            # #sdf_points, sdf_values = sample_sdf_near_surface(mesh, sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)
            # # sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(use_scans=True, sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)

            # combined = np.concatenate((sdf_points, sdf_values[:, np.newaxis]), axis=1)

            combined_surface = np.concatenate((sdf_points_surface, sdf_values_surface[:, np.newaxis]), axis=1)
            combined_uniform = np.concatenate((sdf_points_uniform, sdf_values_uniform[:, np.newaxis]), axis=1)

            np.savetxt(os.path.join(DIRECTORY_SDF_CLOUD,"surface.csv"), combined_surface, delimiter=",")
            np.savetxt(os.path.join(DIRECTORY_SDF_CLOUD,"uniform.csv"), combined_uniform, delimiter=",")
            np.savetxt(os.path.join(DIRECTORY_SDF_CLOUD,"pcd.csv"), pc, delimiter=",")

            # sdf_xyz = sdf_points
            # sdf_gt = sdf_values

            # colors = np.zeros(sdf_xyz.shape)
            # colors[sdf_gt < 0, 2] = 1
            # colors[sdf_gt > 0, 0] = 1

            # pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sdf_xyz))
            # pcd_gt.colors = o3d.utility.Vector3dVector(colors)

            # o3d.visualization.draw_geometries([pcd_gt, line_set])


            #np.save(get_sdf_cloud_filename(filename, DIRECTORY_SDF_CLOUD), combined)
        except BadMeshException:
            tqdm.write("Skipping bad mesh. ({:s})".format(filename))
            # mark_bad_mesh(filename)
            return
        del mesh
            
    except:
        traceback.print_exc()


def process_model_files():
    total_files = []
    sdf_save_dirs = []
    max_obj_size = 5 * 1024 * 1024
    min_obj_size = 5 * 1024
    folders = os.listdir(DATAFILES_FOLDER)
    for _, class_folder in enumerate(folders):
        # if class_folder != '37cfcafe606611d81246538126da07a8':
        #     continue
        save_folder_instance = os.path.join(SAVE_FOLDER, class_folder)
        os.makedirs(save_folder_instance, exist_ok=True)
        print('Processing Class {}'.format(class_folder))
        synset_dir = os.path.join(DATAFILES_FOLDER, class_folder)
        path_to_mesh_model = os.path.join(synset_dir,'output.obj')
        if path.exists(path_to_mesh_model):
            obj_size = Path(path_to_mesh_model).stat().st_size
            if obj_size > max_obj_size:
                #logger.warning('Skipping obj model, too big: %r', uid)
                continue
            if obj_size < min_obj_size:
                #logger.warning('Skipping obj model, too small: %r', uid)
                continue
            total_files.append(path_to_mesh_model)
            sdf_save_dirs.append(save_folder_instance)
            # process_model_file(path_to_mesh_model, save_folder_instance)

    worker_count = 64
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)
    progress = tqdm(total=len(total_files))
    def on_complete(*_):
        progress.update()

    for filename, sdf_dir in zip(total_files, sdf_save_dirs):
        #process_model_file(filename,SDIRECTORY_SDF_CLOUD)
        pool.apply_async(process_model_file, args=(filename,sdf_dir,), callback=on_complete)
    pool.close()
    pool.join()

if __name__ == '__main__':
    process_model_files()