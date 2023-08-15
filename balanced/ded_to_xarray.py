"""

python3 ./balanced/ded_to_xarray.py

"""


import xarray as xar
import dedalus.public as d3

#Load the individual dedalus snapshots into xarray DataAarrays using dedalus' inbuilt 'load_tasks_to_xarray'.
task_list = [d3.load_tasks_to_xarray(f"./balanced/experiments/cyclone_not/balanced_snapshots/balanced_snapshots_s{snap_num+1}.h5") for snap_num in range(50)]

#Create a list of variable names that are stored in these files
list_var_names = [key for key in task_list[0].keys()]

dataset_list = []

#for each variable name, extract the data arrays that contain that data, and then merge them into a single xarray dataset using xarray concat
for var_name in list_var_names:

    list_data_arrays = [task_list_val[var_name] for task_list_val in task_list ]

    dataset_list.append(xar.concat(list_data_arrays, dim='t'))

#Finally, we merge the list of datasets for each variable into a single xarray dataset that contains all variables and all times.
dataset = xar.merge(dataset_list)

# Download dataset
dataset.to_netcdf('./balanced/cyclone_not.nc') 
