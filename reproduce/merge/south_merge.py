"""

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 ./vortices/south_vortices.py
    $ mpiexec -n 4 python3 ./vortices/plot_vortex.py ./vortices/vortex_snapshots/*.h5 --output ./vortices/vortex_frames
    $ mpiexec -n 4 python3 ded_to_xarray.py


To make FFmpeg video:
    $ ffmpeg -r 10 -i ./vortices/vortex_frames/write_%06d.png ./vortices/z_vortex.mp4


mpiexec -n 4 python3 ./reproduce/merge/south_merge.py &&
mpiexec -n 4 python3 ./reproduce/merge/plot_merge.py ./reproduce/merge/merge_snapshots/*.h5 --output ./reproduce/merge/merge_frames &&
ffmpeg -r 40 -i ./reproduce/merge/merge_frames/write_%06d.png ./reproduce/merge/z_merge.mp4

"""


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
Lx, Lz = 1, 1
Nx, Nz = 128, 128
dealias = 3/2                   
stop_sim_time = 100
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

# Set up basic operators
zcross = lambda A: d3.skew(A)

coscolat = dist.Field(name='coscolat', bases=(xbasis, ybasis))
coscolat['g'] = np.cos(np.sqrt((x)**2. + (y)**2) / R)

#--------------------------------------------------------------------------------------------

# INITIAL CONDITIONS


# Initial condition: vortex
#---------------------------

# Parameters
b = 0.5                                              # steepness parameter             
rm = 1e6 * meter                                     # Radius of vortex (km)
vm = 80 * meter / second                             # maximum velocity of vortex

# South pole coordinates
south_lat = [88.6, 83.7, 84.3, 85.0, 84.1, 83.2]
south_long = [211.3, 157.1, 94.3, 13.4, 298.8, 229.7]

# Convert longitude and latitude inputs into x,y coordinates
def conversion(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    return x, y

for i in range(len(south_lat)):

    xx,yy = conversion(south_lat, south_long)
    r = np.sqrt( (x-xx[i])**2 + (y-yy[i])**2 )

    # Overide u,v components in velocity field
    u['g'][0] += vm * ( r / rm ) * np.exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (y-yy[i]) / ( r + 1e-16 ) )
    u['g'][1] += - vm * ( r / rm ) * np.exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (x-xx[i]) / ( r + 1e-16 ) )                          



# Initial condition: height
#---------------------------
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*coscolat*zcross(u))")
problem.add_equation("integ(h) = 0")
solver = problem.build_solver()
solver.solve()


# Initial condition: perturbation
#---------------------------------
# h.fill_random('g')

# pdb.set_trace()

h['g'] = ( np.random.rand(h['g'].shape[0], h['g'].shape[1]) - 0.5 ) * 1.5e-5


#--------------------------------------------------------------------------------------------

# Problem and Solver
#--------------------

# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*coscolat*zcross(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")
# problem.add_equation("integ(h) = 0")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Snapshots
#-----------

# Set up and save snapshots
snapshots = solver.evaluator.add_file_handler('./reproduce/merge/merge_snapshots', sim_dt=0.1, max_writes=10)

# add velocity field
snapshots.add_task(h, name='height')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task((-d3.div(d3.skew(u)) + 2*Omega*coscolat) / (h+H), name='PV')

snapshots.add_task(d3.dot(u,ex), name='u')
snapshots.add_task(d3.dot(u,ey), name='v')
snapshots.add_task(np.sqrt(d3.dot(u,ex)**2 + d3.dot(u,ey)**2), name='vortex')

#--------------------------------------------------------------------------------------------

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)


# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.dot(u,ey)**2, name='w2')


# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()