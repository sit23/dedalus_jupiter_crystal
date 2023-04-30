import numpy as np
import scipy
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pdb

# Parameters
#------------

# Simulation units
meter = 1 / 69.911e6
hour = 1
second = hour / 3600

# Numerical Parameters
Nphi, Nr = 128, 128
dealias = 3/2                   
stop_sim_time = 20
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Planetary Configurations
R = 69.911e6 * meter           
Omega = 1.76e-4 / second            
nu = 1e5 * meter**2 / second / 32**2   
g = 24.79 * meter / second**2      
H = 5e4 * meter  

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge

# Fields
u = dist.VectorField(coords, name='u', bases=disk)
h = dist.Field(name='h', bases=disk)
tau_u = dist.VectorField(coords, name='tau_u', bases=edge)
tau_h = dist.Field(name='tau_h')

# Substitutions
phi, r = dist.local_grids(disk)
lift = lambda A: d3.Lift(A, disk, -1)