import numpy as np
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)

# Build Coordinates

def radon_coords(L, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ]
    )
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L, 2)
    xy = xy @ trans_matrix.T  # (L*L, 2) @ (2, 2)
    xy = xy.reshape(L, L, 2)
    return xy

def ray_coords_2d(length):
    y   = np.linspace(-1, 1, int(length)).reshape(-1, 1)  # (length, ) -> (length, 1)
    x   = np.zeros_like(y)  # (length, 1)
    xy  = np.concatenate((x, y), axis=-1)  # (length, 2)
    return xy

def ray_coords_3d(length):
    y   = np.linspace(-1, 1, int(length)).reshape(-1, 1)  # (length, ) -> (length, 1)
    x   = np.zeros_like(y)  # (length, 1)
    xy  = np.concatenate((x, y), axis=-1)  # (length, 2)
    xyz = np.concatenate((xy, x), axis=-1)  # (length, 3)
    return xyz

def grid_coordinate_2d(L, NL, W, NW):
    x       = torch.linspace(-L, L, NL)
    y       = torch.linspace(-W, W, NW)
    x, y    = torch.meshgrid(x, y, indexing='ij')  # (NL, NW), (NL, NW)
    xy      = torch.stack([x, y], -1).reshape(-1, 2)  # (NL*NW, 2)
    return xy

def grid_coordinate_3d(L, NL, W, NW, H, NH):
    x       = torch.linspace(-L, L, NL)
    y       = torch.linspace(-W, W, NW)
    z       = torch.linspace(-H, H, NH)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')  # (NL, NW, NH), (NL, NW, NH)
    xyz     = torch.stack([x, y, z], axis=-1).reshape(-1, 3)  # (NL*NW*NH, 3)
    return xyz

# General Ray Transformations

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

def ray_rotate_3d(ray_coords, azmuth, polar): 
    azmuth_rad  =   np.deg2rad(azmuth)
    polar_rad   =   np.deg2rad(polar)
    azmuth_mat  =   np.array(
                            [
                            [np.cos(azmuth_rad),    -np.sin(azmuth_rad),    0],
                            [np.sin(azmuth_rad),    np.cos(azmuth_rad),     0],
                            [0,                     0,                      1]
                            ]
                            )
    polar_mat   =   np.array(
                            [
                            [1,     0,                  0                   ],
                            [0,     np.cos(polar_rad),  -np.sin(polar_rad)  ],
                            [0,     np.sin(polar_rad),  np.cos(polar_rad)   ]
                            ]
                            )
    ray_coords  =   ray_coords @ polar_mat.T
    ray_coords  =   ray_coords @ azmuth_mat.T
    return  ray_coords

#Affines are unoptimized. Rework codebase. 

def ray_affine_x_shift(ray_coords,shift):
    if  ray_coords.shape[-1]    ==  2: #2D coords
        ray_coords_temp =   np.concatenate((ray_coords, np.ones((ray_coords.shape[0],1))), axis=-1) # (length, 3)
        mat =   np.array(
                        [
                        [1,     0,      shift   ],
                        [0,     1,      0       ],
                        [0,     0,      1       ]
                        ]
                        )
        ray_coords_temp =   ray_coords_temp @ mat.T
        ray_coords      =   ray_coords_temp[:,0:2].reshape((ray_coords.shape[0],2))
        return  ray_coords
    
    elif  ray_coords.shape[-1]    ==  3: #3D coords
        ray_coords_temp =   np.concatenate((ray_coords, np.ones((ray_coords.shape[0],1))), axis=-1) # (length, 4)
        mat =   np.array(
                        [
                        [1,     0,      0,      shift   ],
                        [0,     1,      0,      0       ],
                        [0,     0,      1,      0       ],
                        [0,     0,      0,      1       ]
                        ]
                        )
        ray_coords_temp =   ray_coords_temp @ mat.T
        ray_coords      =   ray_coords_temp[:,0:3].reshape((ray_coords.shape[0],3))
        return  ray_coords

def ray_affine_y_shift(ray_coords,shift):
    if  ray_coords.shape[-1]    ==  2: #2D coords
        ray_coords_temp = np.concatenate((ray_coords, np.ones((ray_coords.shape[0],1))), axis=-1)   # (length, 3)
        mat =   np.array(
                        [
                        [1,     0,      0       ],
                        [0,     1,      shift   ],
                        [0,     0,      1       ]
                        ]
                        )   
        ray_coords_temp =   ray_coords_temp @ mat.T
        ray_coords      =   ray_coords_temp[:,0:2].reshape((ray_coords.shape[0],2))
        return  ray_coords
    
    elif  ray_coords.shape[-1]    ==  3: #3D coords
        ray_coords_temp = np.concatenate((ray_coords, np.ones((ray_coords.shape[0],1))), axis=-1)   # (length, 4)
        mat =   np.array(
                        [
                        [1,     0,      0,      0       ],
                        [0,     1,      0,      shift   ],
                        [0,     0,      1,      0       ],
                        [0,     0,      0,      1       ]
                        ]
                        )
        ray_coords_temp =   ray_coords_temp @ mat.T
        ray_coords      =   ray_coords_temp[:,0:3].reshape((ray_coords.shape[0],3))
        return  ray_coords

def ray_affine_z_shift(ray_coords,shift):
    ray_coords_temp = np.concatenate((ray_coords, np.ones((ray_coords.shape[0],1))), axis=-1)   # (length, 4)
    mat =   np.array(
                    [
                    [1,     0,      0,      0       ],
                    [0,     1,      0,      0       ],
                    [0,     0,      1,      shift   ],
                    [0,     0,      0,      1       ]
                    ]
                    )
    ray_coords_temp =   ray_coords_temp @ mat.T
    ray_coords      =   ray_coords_temp[:,0:3].reshape((ray_coords.shape[0],3))
    return  ray_coords

# Custom Transforms

# The Radon portion of the code base is the most stable and well tested. 
# This custom geometric/topological transformation section is to exploit the stable codebase 
# and create other geometric/topological structured data consistent with the Radon grid data structure.

# This fanbeam generator is also much more optimized than typical affine ray generators. 
# This operates at O(n), while ray affine generators are O(n^2), though the performance boost literally doesn't matter.

# CBs will just be the 3D version of this. For UNC students and faculty only, contact me if you cannot understand this. 
# I will create a demonstrator Jupyter code to explain this. This comment is dated 01/16/2026. 
# Check back about in a week or 2 for Jupyter demonstrator. 

def fanbeam_geometric_transform(ray_coords, unit_tangent, L, SOD, ODD):
    x_max   =   (L-1)/2
    slope   =   unit_tangent*x_max
    vec_transform   =   ray_coords[:,:,1]*slope + slope*SOD/ODD
    ray_coords[:,:,0] =   np.multiply(vec_transform,ray_coords[:,:,0])
    return ray_coords