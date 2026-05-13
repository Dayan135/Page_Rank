import torch

# This is the neutral ground
_PBR_GPU_CACHE = {}

def get_pbr_gpu_meta(pbr_mat):
    return _PBR_GPU_CACHE.get(id(pbr_mat))

def set_pbr_gpu_meta(pbr_mat, meta_dict):
    _PBR_GPU_CACHE[id(pbr_mat)] = meta_dict

def clear_pbr_gpu_meta(pbr_mat):
    obj_id = id(pbr_mat)
    if obj_id in _PBR_GPU_CACHE:
        del _PBR_GPU_CACHE[obj_id]