import matplotlib.pyplot as plt 
import numpy as np 
import yt 
import sys 
import json 

if __name__=="__main__":
    yt.set_log_level(50)
    on_branch = sys.argv[1]

    ds = yt.load_sample("FIRE_M12i_ref11") 
    _, c = ds.find_max(('gas', 'density'))
    
    dm_widths = range(25,500,25)
    results = {}
    for buff_size_N in (200, 400, 600, 800):
        print(f"running projections for buff_size ({buff_size_N},{buff_size_N})")
        mass_values = []
        for dm_width in dm_widths:
            prj = yt.ProjectionPlot(ds, 'x', ('gas', 'density'), 
                              width=(dm_width, 'Mpc'), 
                              center =c,
                              buff_size=(buff_size_N,)*2)
            frb = prj.frb
            dx = prj.width[0]/prj.buff_size[0]
            dy = prj.width[1]/prj.buff_size[1]
            pixel_area = dx*dy
            inferred_mass = (frb['gas', 'density']*pixel_area).sum()
            mass_values.append(float(inferred_mass.to('g').d))
        
        plt.semilogy(dm_widths, mass_values, marker='.', label=f"{on_branch}, {buff_size_N}")
        plt.xlabel('domain width, Mpc')
        plt.ylabel('inferred total mass, g')
        results[buff_size_N] = mass_values
    plt.legend()
    plt.savefig(f"inferred-mass-on-{on_branch}.png")

    with open(f"inferred-mass-on-{on_branch}.json", 'w') as f:
        json.dump(results, f) 

