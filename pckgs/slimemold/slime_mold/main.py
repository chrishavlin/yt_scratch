import numpy as np 
import numpy.typing as npt
import numba


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
        bins_i = np.clip(bins_i, 0, resolution[idim]-1)
        bins_ids.append(bins_i)
                    

    bins_ids = np.column_stack(bins_ids)
    return bins_ids


@numba.njit
def _find_local_max(positions: npt.NDArray, 
                    bin_ids: npt.NDArray, 
                    bin_positions: npt.NDArray,
                    sense_bins: npt.NDArray,                     
                    deposits:npt.NDArray, 
                    ):

    n_particles = positions.shape[0]

    nx = deposits.shape[1]
    ny = deposits.shape[0] 

    local_max = np.zeros(positions.shape)

    
    for i_particle in range(n_particles):
        
        bin0 = bin_ids[i_particle, 0] # bin number in x, y, z
        bin1 = bin_ids[i_particle, 1] # bin number in x, y, z

        max_val = -100.
        max_pos = [0., 0.]
        for i0 in range(sense_bins[0][0], sense_bins[0,1]):
            for i1 in range(sense_bins[1][0], sense_bins[1,1]):

                ix = bin0 + i0 
                if ix < 0:
                    ix = nx - 1 
                if ix >= nx:
                    ix = 0 

                iy = bin1 + i1
                if iy < 0:
                    iy = ny - 1 
                if iy >= ny:
                    iy = 0 

                val = deposits[iy, ix]
                if val > max_val:
                    max_val = val 
                    max_pos = [bin_positions[0][ix], bin_positions[0][iy]] 

        local_max[i_particle, 0] = max_pos[0]
        local_max[i_particle, 1] = max_pos[1]
                
    return local_max


def _get_particle_directions(positions, 
                             deposits, 
                             position_bins, 
                             bin_ids, 
                             sense_bins,
                             directions=None, 
                             bias = 0.8):    
    resolution = deposits.shape
    ndim = len(resolution)

    if directions is None: 
        directions = np.random.random(positions.shape)
        dir_mag = np.linalg.norm(directions,axis=1)
        for idim in range(ndim):
            directions[:,idim] = directions[:,idim] / dir_mag        

    # global max location                      
    max_loc = _find_local_max(positions, bin_ids, position_bins, sense_bins, deposits)

    # find new direction
    new_direction = positions - max_loc    
    dir_mag = np.linalg.norm(new_direction,axis=1)
    for idim in range(ndim):
        new_direction[:,idim] = new_direction[:,idim] / dir_mag  

    # the bias will depend on where its initially headed 
    # if new_direction = direction, bias is max bias
    bias = bias * (1. - np.linalg.norm(new_direction - directions, axis=1)) 
    
    bias_m1 = 1 - bias
    for idim in range(ndim):
        directions[:, idim] =  bias * new_direction[:,idim] + bias_m1 * directions[:,idim]
       
    # make sure we have unit vector still     
    dir_mag = np.linalg.norm(directions,axis=1)
    for idim in range(ndim):
        directions[:,idim] = directions[:,idim] / dir_mag        

    return directions


def _apply_periodic_bounds(positions, left_edge, right_edge):
    ndim = len(left_edge)
    for idim in range(ndim):
        mask = positions[:,idim] > right_edge[idim]
        wid = right_edge[idim] - left_edge[idim]
        positions[:, idim][mask] = positions[:, idim][mask] - wid
        mask = positions[:,idim] < left_edge[idim]
        positions[:, idim][mask] = positions[:, idim][mask] + wid
    return positions

    
@numba.njit
def diffusion_operator_2d(f: npt.NDArray, nx: int, ny: int, dx: float, dy: float) -> npt.NDArray:

    dfdt = np.zeros(f.shape)
    dx2 = dx * dx
    dy2 = dy * dy
    
    # interior
    for ix in range(1, nx-1):
        for iy in range(1, ny-1):
            dfdt[iy, ix] = (f[iy, ix+1] + f[iy, ix-1] - 2. * f[iy, ix]) / dx2
            dfdt[iy, ix] = dfdt[iy, ix] + (f[iy+1, ix] + f[iy-1, ix] - 2. * f[iy, ix]) / dy2

        # top/bottom
        iy = 0
        dfdt[iy, ix] = (f[iy, ix+1] + f[iy, ix-1] - 2. * f[iy, ix]) / dx2
        dfdt[iy, ix] = dfdt[iy, ix] + (f[iy+1, ix] + f[ny-1, ix] - 2. * f[iy, ix]) / dy2

        iy = ny-1
        dfdt[iy, ix] = (f[iy, ix+1] + f[iy, ix-1] - 2. * f[iy, ix]) / dx2
        dfdt[iy, ix] = dfdt[iy, ix] + (f[0, ix] + f[iy-1,ix] - 2. * f[iy, ix]) / dy2

    
    # left/right
    for iy in range(1, ny-1):
        ix = 0
        dfdt[iy, ix] = (f[iy, ix+1] + f[iy, nx-1] - 2. * f[iy, ix]) / dx2
        dfdt[iy, ix] = dfdt[iy, ix] + (f[iy+1, ix] + f[iy-1, ix] - 2. * f[iy, ix]) / dy2

        ix = nx - 1
        dfdt[iy, ix] = (f[iy, 0] + f[iy, ix-1] - 2. * f[iy, ix]) / dx2
        dfdt[iy, ix] = dfdt[iy, ix] + (f[iy+1, ix] + f[iy-1, ix] - 2. * f[iy, ix]) / dy2


    # corners 
    ix = 0 
    iy = 0
    dfdt[iy, ix] = (f[iy, ix+1] + f[iy, nx-1] - 2. * f[iy, ix]) / dx2
    dfdt[iy, ix] = dfdt[iy, ix] + (f[iy+1, ix] + f[ny-1, ix] - 2. * f[iy, ix]) / dy2

    ix = 0 
    iy = ny - 1
    dfdt[iy, ix] = (f[iy, ix+1] + f[iy, nx-1] - 2. * f[iy, ix]) / dx2
    dfdt[iy, ix] = dfdt[iy, ix] + (f[0, ix] + f[iy-1,ix] - 2. * f[iy, ix]) / dy2

    ix = nx - 1 
    iy = 0 
    dfdt[iy, ix] = (f[iy, 0] + f[iy, ix-1] - 2. * f[iy, ix]) / dx2
    dfdt[iy, ix] = dfdt[iy, ix] + (f[iy+1, ix] + f[ny-1, ix] - 2. * f[iy, ix]) / dy2

    ix = nx - 1 
    iy = ny - 1 
    dfdt[iy, ix] = (f[iy, 0] + f[iy, ix-1] - 2. * f[iy, ix]) / dx2
    dfdt[iy, ix] = dfdt[iy, ix] + (f[0, ix] + f[iy-1,ix] - 2. * f[iy, ix]) / dy2

    return dfdt


@numba.njit
def diffusion_operator_3d(f: npt.NDArray, nx: int, ny: int, dx: float, dy: float) -> npt.NDArray:

    dfdt = np.zeros(f.shape)
    dx2 = dx * dx
    dy2 = dy * dy
    
    for ix in range(1, nx-1):
        dfdt[ix, iy] = (f[ix+1, iy] + f[ix-1, iy] - 2. * f[ix, iy]) / dx2
        for iy in range(1, ny-1):
            dfdt[ix, iy] = dfdt[ix, iy] + (f[ix, iy+1] + f[ix, iy-1] - 2. * f[ix, iy]) / dy2

    return dfdt

            

class MoldSimulation:
    def __init__(self, 
                 resolution = (400, 400),
                 left_edge = (0., 0.),
                 right_edge = (1., 1.), 
                 sense_dist = (0.05, 0.05),
                 n_particles = 100, 
                 deposition_factor = 0.1,
                 diffusivity = 1.0,
                 init_deposit = None):
        self.resolution = resolution
        self.deposition_factor = deposition_factor
        self.diffusivity = diffusivity
        self.n_dim = len(resolution)        
        self.traces = np.zeros(resolution, dtype=float)
        self.trace_i = np.zeros(resolution, dtype=float)

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


        self.sense_dist = np.asarray(sense_dist)
        sense_bins = (np.ceil(self.sense_dist / self.dxyz) / 2 ).astype(int)        
        sense_bins_by_dim = []
        for idim in range(self.n_dim):
            sense_bins_by_dim.append([-sense_bins[idim], sense_bins[idim]])
        self.sense_bins = np.array(sense_bins_by_dim, dtype=int)
        
        self.particle_direction = _get_particle_directions(self.positions, self.deposits, self.position_bins, self.sense_bins, bin_ids, directions=None)
        

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

        if self.n_dim == 2: 
            dxy = [(self.right_edge[idim] - self.left_edge[idim])/self.resolution[idim] for idim in range(2)]
            res = self.resolution
            dDdt = diffusion_operator_2d(self.deposits, res[0], res[1], dxy[0], dxy[1])
        else:
            raise NotImplementedError()
        
        k = self.diffusivity
        dt = np.min(dxy) ** 2 / (2. * k) * 0.5        
        self.deposits = self.deposits + dDdt * k * dt

        
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
        self.particle_direction = _get_particle_directions(self.positions, self.deposits, self.position_bins,  self.sense_bins, bin_ids, directions=self.particle_direction)
    



        

        


       

      


        
        
            
        

    



            
        


