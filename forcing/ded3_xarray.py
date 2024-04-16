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
        dataset = xar.concat(dataset_list, dim='t')

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
    exp_name = 'example_intruder_phys_no_forcing_vortex_sphere_no_noise_low_res33'

    dataset = convert_to_netcdf(exp_name, force_recalculate=False)

    print('plotting')
    plt.figure()
    dataset['vorticity'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')

    plt.figure()
    dataset['ucomp'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')

    plt.figure()
    dataset['PV'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')

    plt.figure()
    dataset['PV'][-1,...].mean('lon').plot.line()    