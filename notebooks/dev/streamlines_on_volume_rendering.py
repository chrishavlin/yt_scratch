import yt
from yt.units import Mpc
from yt.visualization.api import Streamlines
from yt.visualization.volume_rendering.api import LineSource
import numpy as np 

# streamline alpha value: determines opacity of LineSource
# objects. A bit of trial and error to pick a good value 
# so that both the volume rendering and the lines show up.
# this value works for the transfer function used below.
alpha = 0.01 

# Load the dataset
ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")

# Define c: the center of the box, N: the number of streamlines,
# scale: the spatial scale of the streamlines relative to the boxsize,
# and then pos: the random positions of the streamlines.
c = ds.domain_center
N = 10
scale = ds.domain_width[0]
pos_dx = np.random.random((N, 3)) * scale - scale / 2.0
pos = c + pos_dx

# Create streamlines of the 3D vector velocity and integrate them through
# the box defined above
streamlines = Streamlines(
    ds,
    pos,
    ("gas", "velocity_x"),
    ("gas", "velocity_y"),
    ("gas", "velocity_z"),
    length=1.0 * Mpc,
    get_magnitude=True,
)
streamlines.integrate_through_volume()

###############################################################################
# Coercing streamlines for creating LineSource objects
#
# LineSource expects expects line segments... this is not memory efficient, but 
# we'll explode each streamline into a series of line segments. The
# following function takes the positions of a single streamline and resamples
# and reshapes to turn an array of line segments.

def segment_single_streamline(pos_i):
    index_range = np.arange(0, pos_i.shape[0])
    line_indices = np.column_stack([index_range, index_range]).ravel()[1:-1]

    line_segments = pos_i[line_indices, :]
    n_line_segments = int(line_segments.size/6)
    return line_segments.reshape((n_line_segments, 2, 3))

# for example
pos = streamlines.streamlines
line_segments = segment_single_streamline(pos[0])
print(line_segments.shape)

# note the starting point of second position is the end point of first position
print(line_segments[0])
print(line_segments[1])

sc = yt.create_scene(ds)

# for each streamline, expand into line segments and
# add a LineSource
for sid in range(streamlines.streamlines.shape[0]):        
    line_segments = segment_single_streamline(pos[sid])    
    colors = np.ones([line_segments.shape[0], 4])    
    colors[:, -1] = alpha
    lines = LineSource(line_segments, colors)    
    sc.add_source(lines)

sc.save(sigma_clip=4.0)
