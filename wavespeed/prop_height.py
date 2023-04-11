import numpy as np
import xarray as xar
import dedalus.public as d3
import matplotlib.pyplot as plt
import pdb

# Read netCDF file
ds = xar.open_dataset('./wavespeed/H0p9e4_0.01.nc')

# Restrict dataset to x=0 and y=-0.5
height_values = ds.pheight.sel(y=-0.5, x=0)


# Create turning point function that finds the locations of turning points
def turning_points(list):
    dx = np.diff(list)                              # difference not differentiation
    return np.where(dx[1:] * dx[:-1] < 0)           # tuple of array of turning points

# First turning point index location
first_tp = turning_points(height_values)[0][0]

# Location of first maximum point
dt = height_values.t.data[first_tp]
dy = - ds.pheight.y.data[0]

prop_speed = dy / dt
print('dy/dt =', prop_speed)
print('dt =', dt)