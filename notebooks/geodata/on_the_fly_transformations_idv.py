import numpy as np
import yt
import aglio
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import yt_idv

def get_test_array(r, lat, lon):

    r_g, lat_g, lon_g = np.meshgrid(r, lat , lon, indexing='ij')

    freq = 20 * (1 - .8 * (r_g - r.min())/(r.max()-r.min()))


    vals = np.sin((lat_g - lat.mean()) * np.pi/180. * 2 * np.pi * freq)
    vals += np.sin((lon_g - lon.mean()) * np.pi/180. * 2 * np.pi * freq)
    vals = vals * (1 - 0.5 * np.abs(lat_g - lat.mean()) / (lat.max() - lat.min()))
    vals = vals * (1 - 0.5 * np.abs(lon_g - lon.mean()) / (lon.max() - lon.min()))

    return r_g, lat_g, lon_g, np.abs(vals)

r = np.linspace(6371.-2000, 6371, 100)
lat = np.linspace(20, 30, 80)
lon = np.linspace(360-130, 360-120, 70)




r_g, lat_g, lon_g, field_vals = get_test_array(r, lat, lon)


from typing import List


bbox = [[r.min(), r.max()], [lat.min(), lat.max()], [lon.min(), lon.max()]]

def normalize_coord(dim_vals, dim_index):
    return (dim_vals - bbox[dim_index][0]) / (bbox[dim_index][1] -  bbox[dim_index][0])


def get_kdtree(coord_arrays: List[np.ndarray], **kwargs):
    n_dims = len(coord_arrays)
    normalized_coords = []
    for idim in range(n_dims):
        assert coord_arrays[idim].shape == coord_arrays[0].shape

    for idim in range(n_dims):
        dim_1d = coord_arrays[idim].ravel()
        normalized_coords.append(normalize_coord(dim_1d, idim))


    normalized_coords = np.column_stack(normalized_coords)
    return cKDTree(normalized_coords, **kwargs)

field_vals_1d = field_vals.ravel()
the_tree = get_kdtree([r_g, lat_g, lon_g,])

def _mask_outside_bounds(r, lat, lon):
    inside = r > bbox[0][0]
    inside = (inside) & (r < bbox[0][1])
    inside = (inside) & (lat < bbox[1][1])
    inside = (inside) & (lat > bbox[1][0])
    inside = (inside) & (lon < bbox[2][1])
    inside = (inside) & (lon > bbox[2][0])
    return ~inside

def sample_field(field_name, x, y, z):

    fill_val = 0.0 # np.nan

    orig_shape = x.shape
    print(orig_shape)

    # find native coordinate position
    r, lat, lon = aglio.coordinate_transformations.cart2sphere(x, y, z, geo=True, deg=True)
    r = r.ravel()
    lat = lat.ravel()
    lon = lon.ravel()

    # build bounds mask
    outside = _mask_outside_bounds(r, lat, lon)

    r = normalize_coord(r, 0)
    lat = normalize_coord(lat, 1)
    lon = normalize_coord(lon, 2)

    # query the tree
    dists, indexs = the_tree.query(np.column_stack([r, lat, lon]), k=1)

    # select field values
    indexs[outside] = 0
    vals = field_vals_1d[indexs]
    vals[outside] = fill_val

    return vals.reshape(orig_shape)


xyz_g = aglio.coordinate_transformations.geosphere2cart(lat_g, lon_g, r_g)
bbox_cart = np.array([ [dim.min(), dim.max()] for dim in xyz_g])


def _reader(grid, field_name):
    # grid: a yt grid object
    _, fname = field_name

    # first get the internal yt index ranges
    si = grid.get_global_startindex()
    ei = si + grid.ActiveDimensions

    # get the cartesian points for which we want to sample the field
    x = grid[("index", "x")]
    y = grid[("index", "y")]
    z = grid[("index", "z")]

    vals = sample_field(field_name, x, y, z)

    return vals


data = {'density': _reader}
shp = (128, 128, 128)
ds = yt.load_uniform_grid(data, shp, bbox=bbox_cart, length_unit='km')


rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds, ("stream", "density"), no_ghost=True)
rc.run()
