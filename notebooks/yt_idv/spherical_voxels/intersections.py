import numpy as np
from yt_idv.scene_data import _geometry_utils

# // in yt, phi is the polar angle from (0, 2pi), theta is the azimuthal
# // angle (0, pi). the id_ values below are uniforms that depend on the
# // yt dataset coordinate ordering
class SphericalVoxel:
    # attributes that would be calculated in the geometry, vertex shaders or earlier
    # and passed through the pipeline or set as uniforms
    def __init__(self, left_edge, right_edge, axis_id: dict):
        self.left_edge = left_edge
        self.right_edge = right_edge

        phi_planes = self._calculate_phi_planes(left_edge, right_edge, axis_id)
        self.phi_plane_le = phi_planes[0,:]
        self.phi_plane_re = phi_planes[1,:]
        self.id_r = axis_id["r"]
        self.id_theta = axis_id["theta"]
        self.id_phi = axis_id["phi"]

    def _calculate_phi_planes(self, left_edge, right_edge, axis_id):
        edge_coordinates = np.array([left_edge, right_edge])
        return _geometry_utils.phi_normal_planes(edge_coordinates, axis_id)


def get_ray_cone_intersection(theta: float, ray_origin: np.ndarray, ray_dir: np.ndarray):


    # float costheta;
    # vec3 vhat;
    if theta > np.pi/2.0:
        # // if theta is past PI/2, the cone will point in negative z and the
        # // half angle should be measured from the -z axis, not +z.
        vhat = np.ndarray((0.0, 0.0, -1.0))
        costheta = np.cos(np.pi - theta)

    else:
        vhat = np.array((0.0, 0.0, 1.0))
        costheta = np.cos(theta)

    cos2t = costheta ** 2
    # // note: theta = PI/2.0 is well defined. determinate = 0 in that case and
    # // the cone becomes a plane in x-y.

    dir_dot_vhat = np.dot(ray_dir, vhat)
    dir_dot_dir = np.dot(ray_dir, ray_dir)
    ro_dot_vhat = np.dot(ray_origin, vhat)
    ro_dot_dir = np.dot(ray_origin, ray_dir)
    ro_dot_ro = np.dot(ray_origin, ray_dir)

    a_2 = 2.0*(dir_dot_vhat ** 2 - dir_dot_dir * cos2t)
    b = 2.0 * (ro_dot_vhat * dir_dot_vhat - ro_dot_dir*cos2t)
    c = ro_dot_vhat ** 2 - ro_dot_ro*cos2t;
    determinate = b*b - 2.0 * a_2 * c;
    if determinate < 0.0:
        return np.array([np.inf, np.inf])
    elif (determinate == 0.0):
        return np.array([-b / a_2, np.inf])
    else:
        # // note: it is possible to have real solutions that intersect the shadow cone
        # // and not the actual cone. those values should be discarded. But they will
        # // fail subsequent bounds checks for interesecting the volume, so we can
        # // just handle it there instead of adding another check here.
        return np.array([(-b - np.sqrt(determinate))/ a_2, (-b + np.sqrt(determinate))/ a_2]);


def get_ray_plane_intersection(p_normal: np.ndarray, p_constant: float, ray_origin: np.ndarray, ray_dir: np.ndarray):

    n_dot_u = np.dot(p_normal, ray_dir)
    n_dot_ro = np.dot(p_normal, ray_origin)
    # // add check for n_dot_u == 0 (ray is parallel to plane)
    if n_dot_u == 0:
        # // the ray is parallel to the plane. there are either no intersections
        # // or infinite intersections. In the second case, we are guaranteed
        # // to intersect one of the other faces, so we can drop this plane.
        return np.inf

    return (p_constant - n_dot_ro) / n_dot_u


def get_ray_sphere_intersection(r: float, ray_origin: np.array, ray_dir: np.array):

    dir_dot_dir = np.dot(ray_dir, ray_dir)
    ro_dot_ro = np.dot(ray_origin, ray_origin)
    dir_dot_ro = np.dot(ray_dir, ray_origin)
    rsq = r * r; #// could be calculated in vertex shader

    a_2 = 2.0 * dir_dot_dir
    b = 2.0 * dir_dot_ro
    c =  ro_dot_ro - rsq
    determinate = b*b - 2.0 * a_2 * c
    cutoff_val = 0.0
    if determinate < cutoff_val:
        return np.array((np.inf, np.inf))
    elif determinate == cutoff_val:
        return np.array((-b / a_2, np.inf))
    else:
        return np.array([(-b - np.sqrt(determinate))/ a_2, (-b + np.sqrt(determinate))/ a_2])


def transform_to_cartesian(v, voxel):
    # // in yt, phi is the polar angle from (0, 2pi), theta is the azimuthal
    # // angle (0, pi). the id_ values below are uniforms that depend on the
    # // yt dataset coordinate ordering
    id_r = voxel.id_r
    id_theta = voxel.id_theta
    id_phi = voxel.id_phi
    return np.array([v[id_r] * np.sin(v[id_theta]) * np.cos(v[id_phi]),
                    v[id_r] * np.sin(v[id_theta]) * np.sin(v[id_phi]),
                    v[id_r] * np.cos(v[id_theta])])

def transform_to_spherical(v, voxel):
    r = np.sqrt(np.dot(v, v))
    phi = np.arctan(v[1] / v[0])
    xy = np.sqrt(v[0]**2 + v[1]**2)
    theta = np.arctan(xy / v[2])
    outvec = np.zeros((3,))
    outvec[voxel.id_r] = r
    outvec[voxel.id_theta] = theta
    outvec[voxel.id_phi] = phi
    return outvec




def find_all_intersections(v_model_spherical: np.ndarray,
                           camera_pos_cartesian: np.ndarray,
                           voxel: SphericalVoxel):

    # extract data that would be in shader pipeline
    id_r = voxel.id_r
    id_theta = voxel.id_theta
    id_phi = voxel.id_phi
    right_edge = voxel.right_edge
    left_edge = voxel.left_edge
    phi_plane_le = voxel.phi_plane_le
    phi_plane_re = voxel.phi_plane_re

    # now onto the actual calculations...
    ray_position = v_model_spherical #// now spherical
    ray_position_xyz = transform_to_cartesian(ray_position, voxel) #// cartesian
    p0 = camera_pos_cartesian #camera_pos.xyz; #// cartesian

    ray_dir = camera_pos_cartesian - ray_position_xyz
    ray_dir_norm = np.linalg.norm(ray_dir)
    ray_dir = -(ray_dir / ray_dir_norm)


    # // intersections
    print(right_edge[id_r])
    print(ray_position_xyz)
    t_sphere_outer = get_ray_sphere_intersection(right_edge[id_r], ray_position_xyz, ray_dir)
    if np.isinf(t_sphere_outer[0]) and np.isinf(t_sphere_outer[1]) :

        # // if there are no intersections with the outer sphere, then there
        # // will be no intersections with the remaining faces and we can stop
        # // looking.
        return

    t_sphere_inner = get_ray_sphere_intersection(left_edge[id_r], ray_position_xyz, ray_dir)
    t_p_1 = get_ray_plane_intersection(phi_plane_le[0:3], phi_plane_le[3], ray_position_xyz, ray_dir)
    t_p_2 = get_ray_plane_intersection(phi_plane_re[0:3], phi_plane_re[3], ray_position_xyz, ray_dir)
    t_cone_outer = get_ray_cone_intersection(right_edge[id_theta], ray_position_xyz, ray_dir)
    t_cone_inner= get_ray_cone_intersection(left_edge[id_theta], ray_position_xyz, ray_dir)

    t_points =  {
        't_sphere_outer': t_sphere_outer,
        't_sphere_inner': t_sphere_inner,
        't_p_1': np.array([t_p_1,]),
        't_p_2': np.array([t_p_2,]),
        't_cone_outer': t_cone_outer,
        't_cone_inner': t_cone_inner,
    }


    intersections = {}
    intersections_sp = {}
    intersects_in_vol = {}
    for shp, interx in t_points.items():
        interxs = []
        interxs_sp = []
        hits = []
        for tval in interx:
            interxyz = p0 + ray_dir * tval
            interx_sp = transform_to_spherical(interxyz, voxel)
            interxs.append(interxyz)
            interxs_sp.append(interx_sp)
            hits.append(np.all(interx_sp>=left_edge) and np.all(interx_sp<=right_edge))
        intersects_in_vol[shp] = hits
        intersections[shp] = interxs
        intersections_sp[shp] = interxs_sp

    return t_points, intersections, intersections_sp, intersects_in_vol
