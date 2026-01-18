import numpy as np
import SimpleITK as sitk
from torch.utils import data
import DemoUtils as utils

class FanbeamTrainData(data.Dataset):
    def __init__(self, theta, sin_path, sample_N, SOD, ODD, voxel_size):
        self.sample_N = sample_N
        angles = np.linspace(0, 360, theta+1)
        self.angles = angles[:len(angles) - 1]
        sin = sitk.GetArrayFromImage(sitk.ReadImage(sin_path))
        num_angles, L = sin.shape
        self.L = L
        SDD = SOD+ODD
        unit_tangent = voxel_size/SDD
        self.rays = []
        self.projections_lines = []
        ray_grid = utils.radon_coords(L=L, angle=0)
        ray_temp = utils.fanbeam_geometric_transform(ray_grid, unit_tangent, L, SOD=SOD, ODD=ODD).reshape(-1,2)
        
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            ray_temp = utils.ray_rotate_2d(ray_temp,self.angles[0]).reshape(L,L,2)
            self.rays.append(ray_temp)

        self.projections_lines = np.array(self.projections_lines)
        self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
    
        ang = self.angles[item]
        projection_l = self.projections_lines[item]
        ray = self.rays[item]
        index = np.random.randint(0, self.L - self.sample_N, size=1)[0]
        projection_l_sample = projection_l[index:index+self.sample_N]
        ray_sample = ray[index:index+self.sample_N]
        ray_sample = utils.ray_rotate_2d(ray_coords=ray_sample,angle=ang)
        return ray_sample, projection_l_sample