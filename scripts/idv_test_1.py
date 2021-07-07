import yt

import yt_idv

#ds = yt.load_sample("IsolatedGalaxy")

#ds = yt.load('cm1_tornado_lofs/budget-test.04400.000000.nc')
ds = yt.load('cm1_tornado_lofs/nc4_cm1_lofs_tornado_test.nc')


rc = yt_idv.render_context(height=400, width=400, gui=True)
#sg = rc.add_scene(ds, "dbz", no_ghost=True)
sg = rc.add_scene(ds, "zvort", no_ghost=True)
rc.run()

