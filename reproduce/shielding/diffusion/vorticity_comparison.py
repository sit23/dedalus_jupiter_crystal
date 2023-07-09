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

# names = ['ds_nu0p1e1', 'ds_nu1e1', 'ds_nu1e2', 'ds_nu1e3', 'ds_nu1e4', 'ds_nu1e5']


ds_nu0p1e1 = xar.open_dataset(files[0])
ds_nu1e1 = xar.open_dataset(files[1])
ds_nu1e2 = xar.open_dataset(files[2])
ds_nu1e3 = xar.open_dataset(files[3])
ds_nu1e4 = xar.open_dataset(files[4])


print(ds_nu1e1.vorticity)

# for i in range(len(files)):

#     vorticity = ds
