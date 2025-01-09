import numpy as np 
import numba
import numpy.typing as npt

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
def diffusion_operator_3d(f: npt.NDArray, 
                          nx: int, 
                          ny: int, 
                          nz: int, 
                          dx: float, 
                          dy: float, 
                          dz: float) -> npt.NDArray:

    dfdt = np.zeros(f.shape)
    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz
    
    # interior
    for ix in range(1, nx-1):
        for iy in range(1, ny-1):
            for iz in range(1, nz-1):
                dfdt[iz, iy, ix] = (f[iz, iy, ix+1] + f[iz, iy, ix-1] - 2. * f[iz, iy, ix]) / dx2
                dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy+1, ix] + f[iz, iy-1, ix] - 2. * f[iz, iy, ix]) / dy2
                dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz+1, iy, ix] + f[iz-1, iy, ix] - 2. * f[iz, iy, ix]) / dz2

            # top/bottom
            iz = 0
            dfdt[iz, iy, ix] = (f[iz, iy, ix+1] + f[iz, iy, ix-1] - 2. * f[iz, iy, ix]) / dx2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy+1, ix] + f[iz, iy-1, ix] - 2. * f[iz, iy, ix]) / dy2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz+1, iy, ix] + f[nz-1, iy, ix] - 2. * f[iz, iy, ix]) / dz2

            iz = nz - 1
            dfdt[iz, iy, ix] = (f[iz, iy, ix+1] + f[iz, iy, ix-1] - 2. * f[iz, iy, ix]) / dx2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy+1, ix] + f[iz, iy-1, ix] - 2. * f[iz, iy, ix]) / dy2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[0, iy, ix] + f[iz, iy, ix] - 2. * f[iz, iy, ix]) / dz2
        
        # front/back        
        for iz in range(1, nz-1):
            iy = 0        
            dfdt[iz, iy, ix] = (f[iz, iy, ix+1] + f[iz, iy, ix-1] - 2. * f[iz, iy, ix]) / dx2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy+1, ix] + f[iz, ny-1, ix] - 2. * f[iz, iy, ix]) / dy2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz+1, iy, ix] + f[iz-1, iy, ix] - 2. * f[iz, iy, ix]) / dz2

            iy = ny-1        
            dfdt[iz, iy, ix] = (f[iz, iy, ix+1] + f[iz, iy, ix-1] - 2. * f[iz, iy, ix]) / dx2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, 0, ix] + f[iz, iy, ix] - 2. * f[iz, iy, ix]) / dy2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz+1, iy, ix] + f[iz-1, iy, ix] - 2. * f[iz, iy, ix]) / dz2

    
    # left/right
    for iy in range(1, ny-1):
        for iz in range(1, nz-1):
            ix = 0
            dfdt[iz, iy, ix] = (f[iz, iy, ix+1] + f[iz, iy, nx-1] - 2. * f[iz, iy, ix]) / dx2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy+1, ix] + f[iz, iy-1, ix] - 2. * f[iz, iy, ix]) / dy2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz+1, iy, ix] + f[iz-1, iy, ix] - 2. * f[iz, iy, ix]) / dz2

            ix = nx - 1
            dfdt[iz, iy, ix] = (f[iz, iy, 0] + f[iz, iy, ix-1] - 2. * f[iz, iy, ix]) / dx2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy+1, ix] + f[iz, iy-1, ix] - 2. * f[iz, iy, ix]) / dy2
            dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz+1, iy, ix] + f[iz-1, iy, ix] - 2. * f[iz, iy, ix]) / dz2


    # corners 
    for z_c in range(0,2):
        for y_c in range(0,2):
            for x_c in range(0,2):
                ix = 0 + (nx-1) * x_c
                iy = 0 + (ny-1) * y_c
                iz = 0 + (nz-1) * z_c

                ix_m1 = (ix - 1) * x_c + (nx - 1) * (1 - x_c)
                ix_p1 = (nx - 1) * x_c + (ix + 1) * (1 - x_c)
                iy_m1 = (iy - 1) * y_c + (ny - 1) * (1 - y_c)
                iy_p1 = (ny - 1) * y_c + (iy + 1) * (1 - y_c)
                iz_m1 = (iz - 1) * z_c + (nz - 1) * (1 - z_c)
                iz_p1 = (nz - 1) * z_c + (iz + 1) * (1 - z_c)

                dfdt[iz, iy, ix] = (f[iz, iy, ix_p1] + f[iz, iy, ix_m1] - 2. * f[iz, iy, ix]) / dx2
                dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz, iy_p1, ix] + f[iz, iy_m1, ix] - 2. * f[iz, iy, ix]) / dy2
                dfdt[iz, iy, ix] = dfdt[iz, iy, ix] + (f[iz_p1, iy, ix] + f[iz_m1, iy, ix] - 2. * f[iz, iy, ix]) / dz2

    return dfdt


@numba.njit
def _find_local_max_2d(positions: npt.NDArray, 
                    directions: npt.NDArray,
                    bin_ids: npt.NDArray, 
                    bin_positions: npt.NDArray,
                    sense_bins: npt.NDArray,                     
                    deposits:npt.NDArray, 
                    viewing_angle_rads: float,
                    ):

    half_viewing_angle = viewing_angle_rads / 2.

    n_particles = positions.shape[0]

    nx = deposits.shape[1]
    ny = deposits.shape[0] 

    local_max = np.zeros(positions.shape)
    
    for i_particle in range(n_particles):
        
        # bin numbers in each dimension for the particle
        bin0 = bin_ids[i_particle, 0] # bin number in x, y, z
        bin1 = bin_ids[i_particle, 1] # bin number in x, y, z

        max_val = -100.
        max_pos = [0., 0.]
        for i0 in range(sense_bins[0][0], sense_bins[0,1]):
            for i1 in range(sense_bins[1][0], sense_bins[1,1]):
                if i0 == 0 and i1 ==0: 
                    continue
                # handle periodicity
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

                # value of this potential max
                val = deposits[iy, ix]

                if val > max_val:
                    # check if its in the viewing angle
                    posx = bin_positions[0][ix]
                    posy = bin_positions[1][iy]

                    dx = posx - positions[i_particle, 0]
                    dy = posy - positions[i_particle, 1]

                    dirx = directions[i_particle, 0]
                    diry = directions[i_particle, 1]

                    dotval = dirx * dx + diry * dy 
                    magdx = (dx * dx + dy * dy)**(0.5)
                    magdir = (dirx * dirx + diry * diry)**(0.5)
                    angle = np.arccos(dotval / (magdx * magdir))
                    if angle <= half_viewing_angle:
                        # store the max value, position of max value
                        max_val = val 
                        max_pos = [posx, posy] 

        if max_val > 0:
            local_max[i_particle, 0] = max_pos[0]
            local_max[i_particle, 1] = max_pos[1]
                
    return local_max

@numba.njit
def _find_local_max_3d(positions: npt.NDArray, 
                    directions: npt.NDArray,
                    bin_ids: npt.NDArray, 
                    bin_positions: npt.NDArray,
                    sense_bins: npt.NDArray,                     
                    deposits:npt.NDArray, 
                    viewing_angle_rads: float,
                    ):

    half_viewing_angle = viewing_angle_rads / 2.

    n_particles = positions.shape[0]

    nx = deposits.shape[1]
    ny = deposits.shape[0] 
    nz = deposits.shape[2] 


    local_max = np.zeros(positions.shape)
    
    for i_particle in range(n_particles):
        
        # bin numbers in each dimension for the particle
        bin0 = bin_ids[i_particle, 0] # bin number in x, y, z
        bin1 = bin_ids[i_particle, 1] # bin number in x, y, z
        bin2 = bin_ids[i_particle, 2] # bin number in x, y, z

        max_val = -100.
        max_pos = [0., 0., 0.]
        for i0 in range(sense_bins[0][0], sense_bins[0,1]):
            for i1 in range(sense_bins[1][0], sense_bins[1,1]):
                for i2 in range(sense_bins[2][0], sense_bins[2,1]):
                    if i0 == 0 and i1 ==0 and i2 == 0: 
                        continue

                    # handle periodicity
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
                    
                    iz = bin2 + i2
                    if iz < 0:
                        iz = nz - 1 
                    if iz >= nz:
                        iz = 0 

                    # value of this potential max
                    val = deposits[iz, iy, ix]

                    if val > max_val:
                        # check if its in the viewing angle
                        posx = bin_positions[0][ix]
                        posy = bin_positions[1][iy]
                        posz = bin_positions[2][iz]

                        dx = posx - positions[i_particle, 0]
                        dy = posy - positions[i_particle, 1]
                        dz = posz - positions[i_particle, 2]

                        dirx = directions[i_particle, 0]
                        diry = directions[i_particle, 1]
                        dirz = directions[i_particle, 2]

                        dotval = dirx * dx + diry * dy + dirz * dz
                        magdx = (dx * dx + dy * dy + dz * dz)**(0.5)
                        magdir = (dirx * dirx + diry * diry + dirz * dirz)**(0.5)
                        angle = np.arccos(dotval / (magdx * magdir))
                        if angle <= half_viewing_angle:
                            # store the max value, position of max value
                            max_val = val 
                            max_pos = [posx, posy, posz] 

        if max_val > 0:
            local_max[i_particle, 0] = max_pos[0]
            local_max[i_particle, 1] = max_pos[1]
            local_max[i_particle, 2] = max_pos[2]
                
    return local_max