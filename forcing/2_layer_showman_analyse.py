import dedalus.public as d3
import xarray as xar
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob
import ded3_xarray as dedxr
import pdb


exp_name = '2_layer_showman_2007_A1_A5_coupled_mk5'

dataset = dedxr.convert_to_netcdf(exp_name, force_recalculate=False)

output_dir = f'frames/{exp_name}'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for layer in [1,2]:

    print('plotting')
    plt.figure()
    dataset[f'vorticity_{layer}'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/final_vorticity_{layer}.pdf')
    plt.close('all')

    plt.figure()
    dataset[f'height_{layer}'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/final_height_{layer}.pdf')
    plt.close('all')

    plt.figure()
    dataset[f'height_forcing_{layer}'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/final_height_forcing_{layer}.pdf')
    plt.close('all')

    # for h_tick in range(dataset['t'].shape[0]):
    #     plt.figure()
    #     dataset['height'][h_tick,...].plot.contourf(levels=30, cmap='RdBu_r')
    #     plt.savefig(f'{output_dir}/{h_tick}_height.pdf')
    #     plt.close('all')

    plt.figure()
    dataset[f'total_height_{layer}'][-1,...].plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/final_total_height_{layer}.pdf')

    plt.close('all')

    plt.figure()
    dataset[f'ucomp_{layer}'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/ubar_over_time_{layer}.pdf')
    plt.close('all')

    plt.figure()
    dataset[f'ucomp_{layer}'].mean('lon').plot.contour(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/ubar_contour_over_time_{layer}.pdf')
    plt.close('all')

    plt.figure()
    dataset[f'height_{layer}'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/hbar_over_time_{layer}.pdf')
    plt.close('all')

    plt.figure()
    dataset[f'height_forcing_{layer}'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
    plt.savefig(f'{output_dir}/h_forcingbar_over_time_{layer}.pdf')
    plt.close('all')

    try:
        plt.figure()
        dataset[f'PV_{layer}'].mean('lon').plot.contourf(levels=30, cmap='RdBu_r')
        plt.savefig(f'{output_dir}/PVbar_over_time_{layer}.pdf')

        plt.close('all')


        plt.figure()
        dataset[f'PV_{layer}'][-1,...].mean('lon').plot.line()    
        plt.savefig(f'{output_dir}/final_PVbar_{layer}.pdf')
        plt.close('all')
    except:
        pass

    plt.figure()
    dataset[f'height_{layer}'][-1,...].mean('lon').plot.line()    
    plt.savefig(f'{output_dir}/final_hbar_{layer}.pdf')
    plt.close('all')

    plt.figure()
    dataset[f'ucomp_{layer}'][-1,...].mean('lon').plot.line()    
    plt.savefig(f'{output_dir}/final_ubar_{layer}.pdf')
    plt.close('all')

    plt.figure()
    for h_tick in range(dataset['t'].shape[0]):
        plt.plot(dataset['lat'].values, dataset[f'height_{layer}'][h_tick,:,0].values)
        plt.savefig(f'{output_dir}/height0_bar_over_time_{layer}.pdf')    

    plt.close('all')
