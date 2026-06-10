import weakref

# GPU metadata cache keyed by id(pbr_mat). CPython reuses id() values once an
# object is collected, so every entry must be purged when its matrix dies —
# otherwise a new matrix allocated at the same address silently inherits the
# old matrix's GPU tensors (wrong graph, wrong sizes, garbage SpMM output).
_PBR_GPU_CACHE = {}


def get_pbr_gpu_meta(pbr_mat):
    obj_id = id(pbr_mat)
    meta = _PBR_GPU_CACHE.get(obj_id)
    if meta is not None and meta['rem_indptr'].numel() != pbr_mat.rows + 1:
        # id() collision with a dead matrix of a different size (finalizer not
        # supported or not yet run) — drop the stale entry.
        del _PBR_GPU_CACHE[obj_id]
        return None
    return meta


def set_pbr_gpu_meta(pbr_mat, meta_dict):
    obj_id = id(pbr_mat)
    _PBR_GPU_CACHE[obj_id] = meta_dict
    try:
        weakref.finalize(pbr_mat, _PBR_GPU_CACHE.pop, obj_id, None)
    except TypeError:
        pass  # not weak-referenceable; the size check in get() is the fallback


def clear_pbr_gpu_meta(pbr_mat):
    _PBR_GPU_CACHE.pop(id(pbr_mat), None)
