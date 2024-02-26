import yt_idv
from yt_xarray.utilities._grid_decomposition import decompose_image_mask, _get_yt_ds
import numpy as np 
import yt

def _image_mask(x, y, z):
    # a boolean mask indicating where there is data.
    c = (0.5,)*3
    dist_min = 0.3
    dist_max = 0.4
    dist = np.sqrt((x - c[0])**2 + (y-c[1])**2 + (z-c[1])**2)
    mask = dist >= dist_min 
    mask = np.logical_and(mask, dist <= dist_max) * x>=0.5
    return mask
    
def image_callable(grid, field_name):
    x = grid['index', 'x'].d
    y = grid['index', 'y'].d
    z = grid['index', 'z'].d
    
    mask = _image_mask(x, y, z)
    if field_name[1] == 'image':
        return mask * 1.0 
    elif field_name[1] == 'density':
        data = np.random.random(mask.shape)
        data[~mask] = 0. # np.nan
        return data
    

Nx = 64
Ny = 64
Nz = 64   
ix1d = np.linspace(0, 1, Nx)
iy1d = np.linspace(0, 1, Ny)
iz1d = np.linspace(0, 1, Nz)
x, y, z = np.meshgrid(ix1d, iy1d, iz1d)
image_mask = _image_mask(x, y, z)

bbox = np.array([[0,1],[0,1],[0,1]])
data_callables = {
    'image': image_callable, 
    'density': image_callable
}

ds = _get_yt_ds(image_mask, data_callables, bbox, min_grid_size=16, refine_by=2, max_iters=400)


rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds, ("stream", "density"), no_ghost=True)
rc.run()
