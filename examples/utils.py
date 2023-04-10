import open3d as o3d
import numpy as np
import torch

points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
])- 0.5
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)


def labeled_sampling(self, f, subsample, pc_size=1024, load_from_path=True):
    if load_from_path:
        f=pd.read_csv(f, sep=',',header=None).values
        f = torch.from_numpy(f)

    print("f", f.shape)
    half = int(subsample / 2) 
    neg_tensor = f[f[:,-1]<0]
    pos_tensor = f[f[:,-1]>0]

    if pos_tensor.shape[0] < half:
        pos_idx = torch.randint(0, pos_tensor.shape[0], (half,))
    else:
        pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

    if neg_tensor.shape[0] < half:
        if neg_tensor.shape[0]==0:
            neg_idx = torch.randperm(pos_tensor.shape[0])[:half] # no neg indices, then just fill with positive samples
        else:
            neg_idx = torch.randint(0, neg_tensor.shape[0], (half,))
    else:
        neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

    pos_sample = pos_tensor[pos_idx]

    if neg_tensor.shape[0]==0:
        neg_sample = pos_tensor[neg_idx]
    else:
        neg_sample = neg_tensor[neg_idx]

    pc = f[f[:,-1]==0][:,:3]
    print("pc all", pc.shape)
    pc_idx = torch.randperm(pc.shape[0])[:pc_size]
    pc = pc[pc_idx]

    samples = torch.cat([pos_sample, neg_sample], 0)

    return pc.float().squeeze(), samples[:,:3].float().squeeze(), samples[:, 3].float().squeeze() # pc, xyz, sdv
