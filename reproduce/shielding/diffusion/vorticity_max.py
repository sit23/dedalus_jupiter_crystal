"""

python3 ./reproduce/shielding/diffusion/vorticity_max.py

"""


import xarray as xar
import matplotlib.pyplot as plt
import numpy as np
import glob

import pdb

# Read all data sets using glob to find pathnames (returns array of strings with path)
files = sorted(glob.glob('./reproduce/shielding/diffusion/*.nc'))

# experiment names in case needed
names = ['nu1e0', 'nu1e1', 'nu1e2', 'nu1e3', 'nu1e4', 'nu1e5']


# Create empty lists ready for loop
vort = []
max_vort = []

for f in files:

    ds = xar.open_dataset(f)

    for i in range(len(ds.t)):

        maxx = ds.vorticity[i].max().values

        max_vort.append(maxx)

    # set t values
    t = ds.t

    # add max vorticity arrays into one big array
    vort.append(max_vort)

    # reset max_vort list for next loop (possibly better method than this?)
    max_vort=[]


#----------------------------------------------------------------------------

# plot all magnitudes

plt.figure()

for i in range(len(files)):

    plt.plot(t, vort[i], label=names[i])


plt.legend()
plt.title('Max vorticity for varying magntiudes of diffusion')
plt.xlabel('time (t)')
plt.ylabel('vorticity')

plt.savefig('./reproduce/shielding/diffusion/max_vort_all.png')


#----------------------------------------------------------------------------

# Plot smaller magnitudes that look more stable

plt.figure()

for i in range(3):

    plt.plot(t, vort[i], label=names[i])


plt.legend()
plt.title('Max vorticity for varying magntiudes of diffusion')
plt.xlabel('time (t)')
plt.ylabel('vorticity')

plt.savefig('./reproduce/shielding/diffusion/max_vort_smaller.png')