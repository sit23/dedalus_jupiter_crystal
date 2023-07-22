"""

mpiexec -n 4 python3 ./wavespeed_redo/ded_to_xarray.py

"""


import xarray as xar
import dedalus.public as d3
import matplotlib.pyplot as plt

#Load the individual dedalus snapshots into xarray DataAarrays using dedalus' inbuilt 'load_tasks_to_xarray'.
task_list = [d3.load_tasks_to_xarray(f"./wavespeed_redo/wavespeed_snapshots/wavespeed_snapshots_s{snap_num+1}.h5") for snap_num in range(10)]

#Create a list of variable names that are stored in these files
list_var_names = [key for key in task_list[0].keys()]

dataset_list = []

#for each variable name, extract the data arrays that contain that data, and then merge them into a single xarray dataset using xarray concat
for var_name in list_var_names:

    list_data_arrays = [task_list_val[var_name] for task_list_val in task_list ]

    dataset_list.append(xar.concat(list_data_arrays, dim='t'))

#Finally, we merge the list of datasets for each variable into a single xarray dataset that contains all variables and all times.
dataset = xar.merge(dataset_list)

plt.figure()
dataset['pheight'][:,64,:].plot.contourf(levels=30, cmap='RdBu_r')
plt.title('Perturbation height field at x=0.0 as a function of y and time.')

plt.savefig('./wavespeed_redo/pheight.png')

#The slope of the lines on this plot will tell us about how fast the gravity wave travels, and therefore check it's working correctly.

# save to net CDF file
dataset.to_netcdf('./wavespeed_redo/wavespeed.nc') 