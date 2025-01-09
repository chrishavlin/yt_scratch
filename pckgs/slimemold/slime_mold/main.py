import numpy as np 
import numpy.typing as npt
from .numba_operators import diffusion_operator_2d, diffusion_operator_3d, _find_local_max_2d, _find_local_max_3d

def _initial_particle_positions(n_particles: int, n_dim: int, left_edge: npt.NDArray, right_edge: npt.NDArray) -> npt.NDArray:
    pos = []
        
    for idim in range(n_dim):
        pos_i = left_edge[idim] + np.random.random(n_particles) * (right_edge[idim] - left_edge[idim])
        pos.append(pos_i)
             
    return np.column_stack(pos)


def _get_position_bins(n_dim, left_edge, right_edge, resolution):
    # resolution = number of cells
    # returns the domain bins    
    bins = [] 
    for idim in range(n_dim):
        n_edges = resolution[idim] + 1 
        bins.append(np.linspace(left_edge[idim], right_edge[idim], n_edges))
    return bins

def _get_bin_ids(n_dim, positions, position_bins, resolution):
    bins_ids = []

    for idim in range(n_dim):
        bins_i = np.digitize(positions[:,idim], position_bins[idim])            
        bins_i = bins_i - 1 
        bins_i = np.clip(bins_i, 0, resolution[idim]-1)
        bins_ids.append(bins_i)
                    
    
    bins_ids = np.column_stack(bins_ids)
    return bins_ids


def _apply_periodic_bounds(positions, left_edge, right_edge):
    ndim = len(left_edge)
    for idim in range(ndim):
        mask = positions[:,idim] > right_edge[idim]
        wid = right_edge[idim] - left_edge[idim]
        positions[:, idim][mask] = positions[:, idim][mask] - wid
        mask = positions[:,idim] < left_edge[idim]
        positions[:, idim][mask] = positions[:, idim][mask] + wid
    return positions
              

def _check_dimensionality(values):
    dims = [len(value) for value in values]
    if len(set(dims)) != 1:
        msg = "resolution, left_edge, right_edge and sense_dist must all have the same "
        msg += "dimensionality, but they did not."
        raise ValueError(msg)
    

class MoldSimulation:
    def __init__(self, 
                 resolution = (400, 400),
                 left_edge = (0., 0.),
                 right_edge = (1., 1.), 
                 sense_dist = (0.05, 0.05),
                 n_particles = 100, 
                 deposition_factor = 0.1,
                 diffusivity = 1.0,
                 direction_bias = 0.8, 
                 viewing_angle_degs = 45,
                 direction_flip_likelihood = 0.02,          
                 init_deposit = None):
        
        _check_dimensionality([resolution, left_edge, right_edge, sense_dist])
        self.n_dim = len(resolution) 
        self.resolution = resolution
        self.deposition_factor = deposition_factor
        self.diffusivity = diffusivity
        self.viewing_angle_degs = viewing_angle_degs
        self.direction_bias = direction_bias
        self.direction_flip_likelihood = direction_flip_likelihood
                        
        self.traces = np.zeros(resolution, dtype=float)
        self.trace_i = np.zeros(resolution, dtype=float)
        self.time = 0.0
        self.timesteps = 0        
        
        if init_deposit is None:
            self.deposits = np.random.random(resolution)
        else:
            self.deposits = init_deposit
        
    
        self.left_edge = np.asarray(left_edge).astype(float)
        self.right_edge = np.asarray(right_edge).astype(float)
        self.position_bins = _get_position_bins(self.n_dim, left_edge, right_edge, resolution)

        self.domain_wid = self.right_edge - self.left_edge
        self.dxyz = self.domain_wid /  np.asarray(self.resolution)
        self.step_size = np.min(self.dxyz)
        
        self.n_particles = n_particles
        
        self.positions = _initial_particle_positions(n_particles, self.n_dim, left_edge, right_edge)  # lagrangian particle positions
        self.particle_ids = np.arange(n_particles)

        # initial step registration
        bin_ids = _get_bin_ids(self.n_dim, self.positions, self.position_bins, self.resolution)
        self._register_positions_in_trace(bin_ids=bin_ids)

        # the sensing distnce for a particle
        self.sense_dist = np.asarray(sense_dist)
        sense_bins = (np.ceil(self.sense_dist / self.dxyz) / 2 ).astype(int)        
        sense_bins_by_dim = []
        for idim in range(self.n_dim):
            sense_bins_by_dim.append([-sense_bins[idim], sense_bins[idim]])
        self.sense_bins = np.array(sense_bins_by_dim, dtype=int)
                
        # initialize particle direction
        self.particle_direction = self._get_particle_directions(                                                            
                                                           bin_ids,                                                            
                                                           directions=None)
        

    def _register_positions_in_trace(self, bin_ids=None):
        if bin_ids is None: 
            bin_ids = _get_bin_ids(self.n_dim, self.positions, self.position_bins, self.resolution)

        # self.trace_i = np.zeros(self.resolution, dtype=float)
        if self.n_dim == 2:
            self.traces[bin_ids[:,0], bin_ids[:,1]] += 1            
            # self.trace_i[bin_ids[:,0], bin_ids[:,1]] = 1
        else:            
            self.traces[bin_ids[:,0], bin_ids[:,1], bin_ids[:,2]] += 1
            # self.trace_i[bin_ids[:,0], bin_ids[:,1], bin_ids[:,2]] = 1

    def _deposit_at_current_pos(self, bin_ids):
        if self.n_dim == 2:
            self.deposits[bin_ids[:,0], bin_ids[:,1]] += self.deposition_factor 
        else:            
            self.deposits[bin_ids[:,0], bin_ids[:,1], bin_ids[:,2]]  += self.deposition_factor

    def _decay(self):
        # diffusion
        dxyz = self.dxyz
        res = self.resolution
        k = self.diffusivity
        if self.n_dim == 2:                         
            dDdt = diffusion_operator_2d(self.deposits, res[0], res[1], dxyz[0], dxyz[1])
            dt = np.min(dxyz) ** 2 / (2. * k) * 0.5 
        else:
            dDdt = diffusion_operator_3d(self.deposits, res[0], res[1], res[2], dxyz[0], dxyz[1], dxyz[2])
            dt = np.min(dxyz) ** 2 / (2. * k) * 0.25
                
        self.time += dt        
        self.deposits = self.deposits + dDdt * k * dt

        
    def _get_particle_directions(self,                                  
                                bin_ids,                              
                                directions=None,                              
                             ):    
    
        

        bias = self.direction_bias
        viewing_angle_rads = self.viewing_angle_degs * np.pi / 180.

        if directions is None: 
            directions = np.random.random(self.positions.shape)
            dir_mag = np.linalg.norm(directions,axis=1)
            for idim in range(self.n_dim):
                directions[:,idim] = directions[:,idim] / dir_mag

        # local max location within viewing angle        
        if self.n_dim == 2:
            max_loc = _find_local_max_2d(self.positions, 
                                    directions, 
                                    bin_ids, 
                                    self.position_bins, 
                                     self.sense_bins, 
                                    self.deposits, 
                                    viewing_angle_rads)
        else:        
            max_loc = _find_local_max_3d(self.positions, 
                                    directions, 
                                    bin_ids, 
                                    self.position_bins, 
                                     self.sense_bins, 
                                    self.deposits, 
                                    viewing_angle_rads)
            
        # find new direction
        new_direction = (max_loc - self.positions)
        dir_mag = np.linalg.norm(new_direction,axis=1)
        for idim in range(self.n_dim):
            new_direction[:,idim] = new_direction[:,idim] / dir_mag      

        
        bias_m1 = 1 - bias
        directions =  bias * new_direction + bias_m1 * directions
                
        # make sure we have unit vector still and apply a random flip        
        dir_mag = np.linalg.norm(directions,axis=1)
        flip_frac = 1. - self.direction_flip_likelihood
        for idim in range(self.n_dim):
            directions[:,idim] = directions[:,idim] / dir_mag
            rand_flip = np.random.random((directions.shape[0],)) > flip_frac
            directions[rand_flip,idim] = -1 * directions[rand_flip, idim]
        
        return directions

    def step(self):
        
        # move forward with current direction 
        self.positions = self.positions + self.particle_direction * self.step_size
        self.positions = _apply_periodic_bounds(self.positions, self.left_edge, self.right_edge)
        
        # register traces
        bin_ids = _get_bin_ids(self.n_dim, self.positions, self.position_bins, self.resolution)        
        self._register_positions_in_trace(bin_ids=bin_ids)

        # deposit at current position
        self._deposit_at_current_pos(bin_ids)

        # decay 
        self._decay()

        # update direction for next time        
        self.particle_direction = self._get_particle_directions(                                                            
                                                           bin_ids,                                                            
                                                           directions=self.particle_direction)
    
        self.timesteps += 1
        


        

        


       

      


        
        
            
        

    



            
        


