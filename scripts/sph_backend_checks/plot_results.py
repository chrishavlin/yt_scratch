import json 
import matplotlib.pyplot as plt 

for branch in ('main', 'sph_proj_backend'):
    with open(f"inferred-mass-on-{branch}.json",'r') as fi: 
        results = json.load(fi)

    dm_widths = range(25,500,25)
    for buff_size, mass in results.items(): 
        plt.semilogy(dm_widths, mass, label=f"{branch}, {buff_size}",marker='.')

    plt.xlabel('domain width [Mpc]')
    plt.ylabel('inferred mass [g]')
    plt.legend()
    plt.savefig('compare-inferred-by-branch.png')
