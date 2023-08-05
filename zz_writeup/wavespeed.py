"""

mpiexec -n 4 python3 ./original_files/ded_to_xarray.py

"""

# Import libraries
import xarray as xar
import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------

# Convert Snapshots to Xarray
#-----------------------------

# Load snapshots into xarray DataArrays using built in dedalus command
task_list = [d3.load_tasks_to_xarray(f"./original_files/snapshots/snapshots_s{snap_num+1}.h5") for snap_num in range(20)]

# List for variable names
list_var_names = [key for key in task_list[0].keys()]

dataset_list = []

# Create xarray for each variable and put into a list
for var_name in list_var_names:

    list_data_arrays = [task_list_val[var_name] for task_list_val in task_list ]

    dataset_list.append(xar.concat(list_data_arrays, dim='t'))

# Merge list of data sets into single xarray
dataset = xar.merge(dataset_list)

plt.figure()
dataset['pheight'][:,64,:].plot.contourf(levels=30, cmap='RdBu_r')
plt.title('Perturbation height field at x=0.0 as a function of y and time.')
plt.savefig('pheight.png')


#----------------------------------------------------------------------------

# Now to find the slope of the line
#-----------------------------------

# Simulation units for the calculations
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Required parameters
g = 9.80616 * meter / second**2
H = 1e4 * meter


# Isolate perturbation height variable
ds_pheight = dataset.pheight

# Find first maximum point on x=0 and RHS of plot
ds_RHS = ds_pheight.sel(x=0, y=0.9921875) # (last value of y)

# Maximum point is located at: (which is the point t=5.41)
ds_RHS[54]

# Change in t and y
dt = ds_RHS[54].t.data
dy = 1

# Calculate true vs numerical error
c_true = np.sqrt(g*H)
c_num = dy / dt

difference = abs(c_true - c_num)

error = ( difference / c_true ) * 100

print(error)






