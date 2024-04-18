import dedalus.public as d3
import xarray as xar
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob
import ded3_xarray as dedxr
import pdb


exp_name = 'showman_2007_A1_mk5'

dataset = dedxr.convert_to_netcdf(exp_name, force_recalculate=False)

output_dir = f'frames/{exp_name}'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

print('plotting')
plt.figure()
dataset['vorticity'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/final_vorticity.pdf')
plt.close('all')

plt.figure()
dataset['height'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/final_height.pdf')
plt.close('all')

plt.figure()
dataset['height_forcing'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/final_height_forcing.pdf')
plt.close('all')

# for h_tick in range(dataset['t'].shape[0]):
#     plt.figure()
#     dataset['height'][h_tick,...].plot.contourf(levels=30, cmap='RdBu_r')
#     plt.savefig(f'{output_dir}/{h_tick}_height.pdf')
#     plt.close('all')

plt.figure()
dataset['total_height'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/final_total_height.pdf')

plt.close('all')

plt.figure()
dataset['ucomp'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/ubar_over_time.pdf')
plt.close('all')

plt.figure()
dataset['ucomp'].mean('lon').plot.contour(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/ubar_contour_over_time.pdf')
plt.close('all')

plt.figure()
dataset['height'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/hbar_over_time.pdf')
plt.close('all')

plt.figure()
dataset['height_forcing'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/h_forcingbar_over_time.pdf')
plt.close('all')

plt.figure()
dataset['PV'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/PVbar_over_time.pdf')

plt.close('all')


plt.figure()
dataset['PV'][-1,...].mean('lon').plot.line()    
plt.savefig(f'{output_dir}/final_PVbar.pdf')
plt.close('all')

plt.figure()
dataset['height'][-1,...].mean('lon').plot.line()    
plt.savefig(f'{output_dir}/final_hbar.pdf')
plt.close('all')

plt.figure()
for h_tick in range(dataset['t'].shape[0]):
    plt.plot(dataset['lat'].values, dataset['height'][h_tick,:,0].values)
    plt.savefig(f'{output_dir}/height0_bar_over_time.pdf')    

plt.close('all')
