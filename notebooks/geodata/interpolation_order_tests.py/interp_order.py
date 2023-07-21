def build_and_select(nx, ny ,nz, method):

    fields = {'temperature': ('x', 'y', 'z'), 
              'pressure': ('x', 'y', 'z')}
    dims = {'x': (0,1,nx), 'y': (0, 1, ny), 'z': (0, 1, nz)}
    ds_xr = load_random_xr_data(fields, dims, length_unit='m')

    # select the whole grid 
    si = np.array([0, 0, 0])
    ei = np.array(ds_xr.temperature.shape) - 1
    
        
    data = select_yt_cell_centers(si, 
                                  ei, 
                                  'temperature',
                                 method=method)
    return data