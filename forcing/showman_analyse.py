import dedalus.public as d3
import xarray as xar
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob
import ded3_xarray as dedxr
import json
import pdb


exp_name = 'showman_2007_A1_mk15'

dataset = dedxr.convert_to_netcdf(exp_name, force_recalculate=False)

with open(f'snapshots/{exp_name}/sim_variables.json', 'r') as f:
    # Load JSON data from file
    params_dict = json.load(f)

grav = params_dict['g']*params_dict['second']**2./params_dict['meter']

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
dataset['ucomp'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
plt.savefig(f'{output_dir}/final_ucomp.pdf')
plt.close('all')

plt.figure()
dataset['height'][-1,...].plot.contourf(levels=30, cmap='gray')
plt.savefig(f'{output_dir}/final_height_BW.pdf')
plt.close('all')

for t_val in [14., 139., 600., 1400., 4190.]:

    plt.figure()
    step=4
    plt.pcolormesh(dataset['lon'], dataset['lat'], dataset['total_height'].sel(t=t_val, method='nearest'), cmap='gray')
    plt.colorbar()
    plt.quiver(dataset['lon'][::step], dataset['lat'][::step], dataset['ucomp'].sel(t=t_val, method='nearest')[::step,::step], dataset['vcomp'].sel(t=t_val, method='nearest')[::step,::step], color='w')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.xlim([0.,120.])
    plt.ylim([0.,70.])
    gh = dataset['total_height'].sel(t=t_val, method='nearest').sel(lon=slice(0.,120.), lat=slice(0.,70.))*grav
    gh_max = gh.max().values
    gh_min = gh.min().values 
    u_both_abs = np.abs(dataset['u'].sel(t=t_val, method='nearest').sel(lon=slice(0.,120.), lat=slice(0.,70.)))
    u_max = u_both_abs.max().values
    plt.title(f'gh ={gh_min/1e5:.2f}-{gh_max/1e5:.2f}x 10^5m^2/s, speed_max = {u_max:.2f}m/s')
    plt.savefig(f'{output_dir}/final_quiver_{step}_t_{t_val}_showman_domain.pdf')
    plt.close('all')

# pdb.set_trace()

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
dataset['ucomp'][-1,...].mean('lon').plot.line()    
plt.savefig(f'{output_dir}/final_ubar.pdf')
plt.close('all')

plt.figure()
for h_tick in range(dataset['t'].shape[0]):
    plt.plot(dataset['lat'].values, dataset['height'][h_tick,:,0].values)
    plt.savefig(f'{output_dir}/height0_bar_over_time.pdf')    

plt.close('all')
