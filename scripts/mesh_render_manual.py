import numpy as np
import yt
import yt_idv



def get_plane_approx(nx,ny,nz):


    x = np.linspace(0, 1, nx) # element-centers
    z = np.linspace(0, 1, nz) # element-centers
    y = np.linspace(0, 1, ny)

    def get_dval(xc,dmin = 0.01):
        if len(xc) > 1:
            return xc[1] - xc[0]
        else:
            return dmin

    def get_edge(xc, dx):
        if len(xc) > 1:
            return np.append([xc - dx/2.], [xc[-1] + dx/2])
        else:
            return np.array([0.5 - dx/2., 0.5 + dx/2.])

    dx = get_dval(x)
    dz = get_dval(z)
    dy = get_dval(y)

    xe = get_edge(x, dx)
    ye = get_edge(y, dy)
    ze = get_edge(z, dz)

    # assemble elements
    coords = []
    connectivity = []
    icoord = 0 

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                verts = [
                            [xe[ix], ye[iy], ze[iz]],
                            [xe[ix+1], ye[iy], ze[iz]],
                            [xe[ix+1], ye[iy+1], ze[iz]],
                            [xe[ix], ye[iy+1], ze[iz]],
                            [xe[ix], ye[iy], ze[iz+1]],
                            [xe[ix+1], ye[iy], ze[iz+1]],
                            [xe[ix+1], ye[iy+1], ze[iz+1]],
                            [xe[ix], ye[iy+1], ze[iz+1]]
                        ]

                connectivity += [[i for i in range(icoord, icoord+8)],]

                coords += verts
                icoord += 8

    coords = np.array(coords)
    connectivity = np.array(connectivity)


    datavals =  coords[:,0][connectivity.ravel()].reshape(connectivity.shape)
    return coords, connectivity, datavals




full_con = []
full_coord = []
full_data = []
offset = 0
for nx,ny,nz in [(1,15,20), (15, 1, 20), ]:
    coord, conn, data = get_plane_approx(nx, ny, nz)

    full_con.append(conn + offset)
    offset += coord.shape[0]
    full_data.append(data)
    full_coord.append(coord)

full_coord = np.concatenate(full_coord)
full_con = np.concatenate(full_con)
full_data = {('connect1','test'): np.concatenate(full_data)}




ds = yt.load_unstructured_mesh(full_con, full_coord, full_data)


rc = yt_idv.render_context(height=800, width=800, gui=True)

from yt_idv.cameras.trackball_camera import TrackballCamera  # NOQA
from yt_idv.scene_components.mesh import MeshRendering  # NOQA
from yt_idv.scene_data.mesh import MeshData  # NOQA
from yt_idv.scene_graph import SceneGraph  # NOQA

c = TrackballCamera(position=[3.5, 3.5, 3.5], focus=[0.0, 0.0, 0.0])
rc.scene = SceneGraph(camera=c)

dd = ds.all_data()
md = MeshData(data_source=dd)
md.add_data(("connect1", "test"))
mr = MeshRendering(data=md, cmap_log=False)

rc.scene.data_objects.append(md)
rc.scene.components.append(mr)
rc.run()

