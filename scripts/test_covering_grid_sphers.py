
from yt.testing import fake_random_ds
import numpy as np

# for nprocs in [1, 2, 4, 8]:
nprocs = 1
level = 2
ds = fake_random_ds(32, nprocs=nprocs)

dn = ds.refine_by**level
dims = dn * ds.domain_dimensions

def get_volume(sp):
    cg_sp = ds.covering_grid(level, [0.0, 0.0, 0.0], dims, data_source=sp)
    sp_mask = cg_sp["gas", "density"] != 0  # values inside sp will be nonzero
    sphere_vol = cg_sp[("index", "cell_volume")] * sp_mask
    return sphere_vol.sum()

vols = []
radii = ds.arr(np.linspace(0.05, 0.45, 30), "code_length")
actual_vols = 4/3 * np.pi * radii**3
for r_ in radii:
    sp_ = ds.sphere(ds.domain_center, r_)
    vols.append(get_volume(sp_))
vols = ds.arr(vols)


import matplotlib.pyplot as plt 

f, axs = plt.subplots(1,2)
axs[0].plot(radii, actual_vols, 'k', label=None)
axs[0].plot(radii, actual_vols, '.k', label="actual volume")
axs[0].plot(radii, vols, 'or',markerfacecolor='none', label="volume from c. grid")
axs[0].set_ylabel('volume')
axs[0].set_xlabel('radius')
axs[0].legend()

axs[1].plot(radii, np.abs(actual_vols - vols)/actual_vols,'.k')
axs[1].plot(radii, np.abs(actual_vols - vols)/actual_vols,'k')
axs[1].set_ylabel('|(actual vol - calculated vol) / actual vol |')
axs[1].set_xlabel('radius')
plt.show()
