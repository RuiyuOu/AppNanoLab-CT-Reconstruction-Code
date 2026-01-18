import numpy as np
import torch

def radon_coords(L, angle):
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    x, y = np.meshgrid(x, y, indexing='ij')
    xy = np.stack([x, y], -1).reshape(-1, 2)
    xy = xy.reshape(L, L, 2)
    return xy

def grid_coordinate_2d(L, NL, W, NW):
    x       = torch.linspace(-L, L, NL)
    y       = torch.linspace(-W, W, NW)
    x, y    = torch.meshgrid(x, y, indexing='ij') 
    xy      = torch.stack([x, y], -1).reshape(-1, 2) 
    return xy

def ray_rotate_2d(ray_coords, angle):
    angle_rad   =   np.deg2rad(angle)
    mat         =   np.array(
                            [
                            [np.cos(angle_rad),     -np.sin(angle_rad)],
                            [np.sin(angle_rad),     np.cos(angle_rad)],
                            ]
                            )   
    ray_coords  =   ray_coords @ mat.T
    return  ray_coords

def fanbeam_geometric_transform(ray_coords, unit_tangent, L, SOD, ODD):
    x_max   =   (L-1)/2
    slope   =   unit_tangent*x_max
    vec_transform   =   ray_coords[:,:,1]*slope + slope*SOD/ODD
    ray_coords[:,:,0] =   np.multiply(vec_transform,ray_coords[:,:,0])
    return ray_coords