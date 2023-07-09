"""

python3 ./reproduce/shielding/diffusion/vorticity_comparison.py

"""


import xarray as xar
import matplotlib.pyplot as plt
import numpy as np
import glob

import pdb

# Read all data sets using glob to find pathnames (returns array of strings with path)
files = sorted(glob.glob('./reproduce/shielding/diffusion/*.nc'))

names = ['ds_nu0p1e1', 'ds_nu1e1', 'ds_nu1e2', 'ds_nu1e3', 'ds_nu1e4', 'ds_nu1e5']

vort = []

for f in files:

    ds = xar.open_dataset(f)

    vorticity = ds.vorticity[:,256,256]
    t = ds.t

    vort.append(vorticity)

#----------------------------------------------------------------------------

# plot all magnitudes

plt.figure()

for i in range(len(files)):

    plt.plot(t, vort[i], label=names[i])


plt.legend()
plt.title('Vorticity for varying magntiudes of diffusion')
plt.xlabel('time (t)')
plt.ylabel('vorticity')

plt.savefig('./reproduce/shielding/diffusion/plot_vort_all.png')
plt.show()


#----------------------------------------------------------------------------

# Plot small magnitudes

plt.figure()

for i in range(len(files)):

    plt.plot(t, vort[i], label=names[i])


plt.legend()
plt.title('Vorticity for varying magntiudes of diffusion')
plt.xlabel('time (t)')
plt.ylabel('vorticity')

plt.savefig('./reproduce/shielding/diffusion/plot_vort_all.png')
plt.show()