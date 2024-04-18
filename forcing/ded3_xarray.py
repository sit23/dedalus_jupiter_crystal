import dedalus.public as d3
import xarray as xar
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob
import pdb

def convert_to_netcdf(exp_name, force_recalculate=False):

    nc_name = f'snapshots/{exp_name}/combined.nc'

    if not os.path.exists(nc_name) or force_recalculate:
        print('combined nc file does not exist. Calculating')
        files = glob.glob(f"snapshots/{exp_name}/{exp_name}_s*.h5")

        dataset_list = []

        for file in tqdm(files):
            tasks = d3.load_tasks_to_xarray(file)

            task_list = [tasks[task_key] for task_key in tasks.keys()]

            dataset = xar.merge(task_list)

            dataset_list.append(dataset)

        print('concatanating')
        dataset = xar.combine_by_coords(dataset_list)

        deg_lat = np.rad2deg(np.pi / 2 - dataset['theta'])
        deg_lon = np.rad2deg(dataset['phi'])

        dataset = dataset.rename({'theta':'lat', 'phi':'lon'})
        dataset['lat'] = ('lat', deg_lat.values)
        dataset['lon'] = ('lon', deg_lon.values)

        dim_list = [dim for dim in dataset.dims.keys()]
        var_list = [key for key in dataset.keys() if key not in dim_list]

        if '' in dim_list:
            dataset = dataset.rename({'':'comp'})

        dataset = dataset.transpose('t','lat','lon','comp')

        if 'u' in var_list:
            dataset['ucomp'] = (dataset['u'][...,0].dims, dataset['u'][...,0].values)
            dataset['vcomp'] = (dataset['u'][...,1].dims, (-dataset['u'][...,1]).values)        

        print('writing output')
        dataset.to_netcdf(nc_name)
    else:
        print('combined nc file does exist. Loading')
        dataset = xar.open_dataset(nc_name)
    
    return dataset



if __name__=="__main__":
    exp_name = 'phys_forcing_no_crystal_bd_recreate_Ro_100_taurad_0_mk7'

    dataset = convert_to_netcdf(exp_name, force_recalculate=False)

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

    for h_tick in range(dataset['t'].shape[0]):
        plt.figure()
        dataset['height'][h_tick,...].plot.contourf(levels=30, cmap='RdBu_r')
        plt.savefig(f'{output_dir}/{h_tick}_height.pdf')
        plt.close('all')

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
