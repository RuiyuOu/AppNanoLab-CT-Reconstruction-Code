import numpy as np
import SimpleITK as sitk
from torch.utils import data
import utils

class RadonTestData(data.Dataset):
    def __init__(self, theta, L):
        # generate views
        angles = np.linspace(0, 180, theta+1)
        angles = angles[:len(angles) - 1]
        num_angles = len(angles)
        # build parallel rays
        self.rays = []
        for i in range(num_angles):
            self.rays.append(utils.radon_coords(L=L, angle=angles[i]))

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]  # (L, L, 2)
        return ray


class RadonTrainData(data.Dataset):
    def __init__(self, theta, sin_path, sample_N):
        self.sample_N = sample_N
        # generate views
        angles = np.linspace(0, 180, theta+1)
        angles = angles[:len(angles) - 1]
        # load sparse-view sinogram
        sin = sitk.GetArrayFromImage(sitk.ReadImage(sin_path))
        num_angles, L = sin.shape
        # store sparse-view sinogram and build parallel rays
        self.rays = []
        self.projections_lines = []
        for i in range(num_angles):
            self.projections_lines.append(sin[i, :])  # (, L)
            self.rays.append(utils.radon_coords(L=L, angle=angles[i]))

        self.projections_lines = np.array(self.projections_lines)
        self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        # sample view
        projection_l = self.projections_lines[item]     # (L, )
        ray = self.rays[item]   # (L, L, 2)
        # sample ray
        sample_indices = np.random.choice(len(projection_l), self.sample_N, replace=False)
        projection_l_sample = projection_l[sample_indices]  # (sample_N)
        ray_sample = ray[sample_indices]    # (sample_N, L, 2)
        return ray_sample, projection_l_sample
    

class FanbeamTestData(data.Dataset):
    def __init__(self, theta, sin_path, SOD, ODD, voxel_size):
        sin = sitk.GetArrayFromImage(sitk.ReadImage(sin_path))
        num_angles, L = sin.shape
        angles = np.linspace(0, 360, theta+1)
        angles = angles[:len(angles) - 1]
        num_angles = len(angles)
        SDD = SOD+ODD
        unit_tangent = voxel_size/SDD
        self.rays = []
        ray_grid = utils.radon_coords(L=L, angle=0)
        ray_temp = utils.fanbeam_geometric_transform(ray_grid, unit_tangent, L, SOD=SOD, ODD=ODD).reshape(-1,2)
        
        for i in range(num_angles):
            ray_temp = utils.ray_rotate_2d(ray_temp,angles[i]).reshape(L,L,2)
            self.rays.append(ray_temp)

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        ray = self.rays[item]  # (L, L, 2)
        return ray


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
        
        for i in range(num_angles): #unoptimized but mimicks the stable Radon code. The unoptimized part is also irrelevant in the long term run time, which is training, and thus this is kept the way that it is. 
            self.projections_lines.append(sin[i, :])  # (, L)
            ray_temp = utils.ray_rotate_2d(ray_temp,self.angles[0]).reshape(L,L,2)
            self.rays.append(ray_temp)

        self.projections_lines = np.array(self.projections_lines)
        self.rays = np.array(self.rays)

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
    
        ang = self.angles[item]
        # sample view
        projection_l = self.projections_lines[item]     # (L, )
        ray = self.rays[item]   # (L, L, 2)
        # sample ray. different sampling pattern that the Radon grid. 
        index = np.random.randint(0, self.L - self.sample_N, size=1)[0]
        projection_l_sample = projection_l[index:index+self.sample_N]  # (sample_N)
        ray_sample = ray[index:index+self.sample_N]    # (sample_N, L, 2)
        ray_sample = utils.ray_rotate_2d(ray_coords=ray_sample,angle=ang)
        return ray_sample, projection_l_sample