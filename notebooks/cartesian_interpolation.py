import yt
import yt_idv
import numpy as np 
from scipy.interpolate import NearestNDInterpolator


bbox = np.array([[0., 1.], [0.0, 2 * np.pi], [0, np.pi]])
sz = (50, 50, 50)
fake_data = {"density": np.random.random(sz)}

def _neato(field, data):
    r = data["index", "r"].d
    theta = data["index", "theta"].d
    phi = data["index", "phi"].d
    phi_c = 0.25 * np.pi
    theta_c = 0.5 * np.pi

    # decay away from phi_c, theta_c
    fac = np.exp(-(((phi_c - phi) / 0.5) ** 2)) * np.exp(
        -(((theta_c - theta) / 0.5) ** 2)
    )

    # cos^2 variation in r with slight increase towards rmin
    rfac = np.cos((r - 0.1) / 0.9 * 3 * np.pi) ** 2 * (1 - 0.25 * (r - 0.1) / 0.9)
    field = fac * rfac + 0.1 * np.random.random(r.shape)

    # field = field * (theta <= 2.0) * (phi < 1.25)
    return field


yt.add_field(
    name=("stream", "neat"),
    function=_neato,
    sampling_type="local",
    units="",
    force_override=True,
    take_log=False,
)

ds = yt.load_uniform_grid(
    fake_data,
    sz,
    bbox=bbox,
    nprocs=256,
    geometry="spherical",
    axis_order =("r", "phi", "theta"),
    length_unit="m",
)

ad = ds.all_data()
_ = ad[("stream", "neat")] # dumb. need it cause of weird field init

def _get_cartesian_neat(grid, field_name):

    # get the target x, y, z in the new cartesian dataset

    dxyz = ds.quan(0., 'code_length') # might want some slop
    grid_bbox = [ [grid.LeftEdge[idim]-dxyz, grid.RightEdge[idim]+dxyz] for idim in range(3)]

    # select subset of the domain in the original spherical dataset based on current
    # grid range
    conditionals = [f"obj[('index', 'cartesian_x')] <= {grid_bbox[0][1].d}",
                    f"obj[('index', 'cartesian_x')] >= {grid_bbox[0][0].d}",
                    f"obj[('index', 'cartesian_y')] <= {grid_bbox[1][1].d}",
                    f"obj[('index', 'cartesian_y')] >= {grid_bbox[1][0].d}",
                    f"obj[('index', 'cartesian_z')] <= {grid_bbox[2][1].d}",
                    f"obj[('index', 'cartesian_z')] >= {grid_bbox[2][0].d}",
                   ]
    cut_region = ds.cut_region(ds.all_data(), conditionals)
    xi = cut_region[('index', 'cartesian_x')]
    yi = cut_region[('index', 'cartesian_y')]
    zi = cut_region[('index', 'cartesian_z')]
    neat_raw = cut_region[('stream', 'neat')]

    # interpolate!
    print(f"interpolator constructed with {neat_raw.size} points")
    interpolator = NearestNDInterpolator(np.column_stack((xi, yi, zi)), neat_raw)

    x = grid[('index', 'x')]
    y = grid[('index', 'y')]
    z = grid[('index', 'z')]
    new_vals = interpolator(x.ravel(), y.ravel(), z.ravel()).reshape(x.shape)
    return new_vals

def _neato(field, data):
    gn = data[("stream", "interpd_neat")]
    
    x = data[('index', 'x')]
    y = data[('index', 'y')]
    z = data[('index', 'z')]
    r = np.sqrt(x**2 + y**2 + z**2)
    
    gn[r>1.] = 0.0
    return gn 


yt.add_field(
    name=("stream", "neat"),
    function=_neato,
    sampling_type="local",
    units="",
    force_override=True,
    take_log=False,
)



grid_data = [
  dict(
      left_edge=[-1.0, -1.0, -1.0],
      right_edge=[1.0, 1.0, 1.0],
      level=0,
      dimensions=[32, 32, 32],
  ),
     dict(
         left_edge=[0., 0., 0.],
         right_edge=[0.75, 0.75, 0.75],
         level=1,
         dimensions=[32, 32, 32],
     ),
    dict(
         left_edge=[-.25, -0.25, -0.25],
         right_edge=[0.25, 0.25, 0.25],
         level=2,
         dimensions=[32, 32, 32],
     ),
 ]

for g in grid_data:
    g["interpd_neat"] = _get_cartesian_neat


bbox = np.array([[-1, 1], [-1, 1], [-1, 1]])
ds3 = yt.load_amr_grids(grid_data, [32, 32, 32], length_unit=1.0, bbox = bbox)


rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds3, "neat", no_ghost=True)
rc.run()
