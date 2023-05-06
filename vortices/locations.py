import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import pdb

# Simulation units
meter = 1 / 69.911e6
hour = 1
second = hour / 3600

# Numerical Parameters
Lx, Lz = 1, 1
Nx, Nz = 128, 128
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

#--------------------------------------------------------------------------------------------

# Dedalus set ups
#-----------------

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)                                                  
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields both functions of x,y
h = dist.Field(name='h', bases=(xbasis,ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))

# Substitutions
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

#--------------------------------------------------------------------------------------------


# Set up basic operators
zcross = lambda A: d3.skew(A) # 90deg rotation anticlockwise (positive)

coscolat = dist.Field(name='coscolat', bases=(xbasis, ybasis))
coscolat['g'] = np.cos(np.sqrt((x)**2. + (y)**2) / R)                                       # ['g'] is shortcut for full grid


# INITIAL CONDITIONS

# Parameters
b = 1.5                                                # steepness parameter             
rm = 1e6 * meter                                     # Radius of vortex (km)
vm = 80 * meter / second                             # maximum velocity of vortex

a = 0.1
r = np.sqrt((x-a)**2 + (y-a)**2)                     # radius

#--------------------------------------------------------------------------------------------


# PLOTTING

