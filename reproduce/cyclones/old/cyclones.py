"""

python3 ./reproduce/cyclones/delete_cyclones.py &&
mpiexec -n 4 python3 ./reproduce/cyclones/cyclones.py &&
mpiexec -n 4 python3 ./reproduce/cyclones/plot_cyclones.py ./reproduce/cyclones/cyclones_snapshots/*.h5 --output ./reproduce/cyclones/cyclones_frames &&
ffmpeg -r 50 -i ./reproduce/cyclones/cyclones_frames/write_%06d.png ./reproduce/cyclones/z_cyclones.mp4

Stitching two mp4s together:
    - ffmpeg -i ./reproduce/cyclones/z_cyclones1.mp4 -i ./reproduce/cyclones/z_anticyclones1.mp4 -filter_complex hstack ./reproduce/cyclones/all_cyclones.mp4

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
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Length of simulation
days = 40
stop_sim_time = 24 * days
printout = 1

# Planetary Configurations
R = 69.911e6 * meter           
Omega = 1.76e-4 / second            
nu = 1e5 * meter**2 / second / 32**2   
g = 24.79 * meter / second**2           


#-----------------------------------------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------------------------------------

# INITIAL CONDITIONS

# Independent variables
#-----------------------

# Steepness parameter
b = 1.5

# Rossby Number
Ro = 0.2

# Dependent variables
#---------------------

# Calculate max speed with Rossby Number
f0 = 2 * Omega                                       # Planetary vorticity
rm = 1e6 * meter                                     # Radius of vortex (km)
vm = Ro * f0 * rm                                    # Calculate speed with Ro

# Calculate deformation radius with Burger number
H = 5e4 * meter
phi = g * (h + H)

# Burger Number -- Currently Bu ~ 10
# Bu = phi / (f0 * rm)**2 
# pdb.set_trace()


# Initial condition: off-centre cyclone
#---------------------------------------

a = 0.25
r = np.sqrt((x-a)**2 + (y-a)**2)                     # radius

# Overide u,v components in velocity field
u['g'][0] = - vm * ( r / rm ) * np.exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (y-a) / ( r + 1e-16 ) )
u['g'][1] = vm * ( r / rm ) * np.exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (x-a) / ( r + 1e-16 ) )                         



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
h['g'] += ( np.random.rand(h['g'].shape[0], h['g'].shape[1]) - 0.5 ) * 1e-6


#-----------------------------------------------------------------------------------------------------------------

# Problem and Solver
#--------------------

# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*coscolat*zcross(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Snapshots
#-----------

# Set up and save snapshots
snapshots = solver.evaluator.add_file_handler('./reproduce/cyclones/cyclones_snapshots', sim_dt=printout, max_writes=10)

# add velocity field
snapshots.add_task(h, name='height')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task((-d3.div(d3.skew(u)) + 2*Omega*coscolat) / phi, name='PV')

snapshots.add_task(d3.dot(u,ex), name='u')
snapshots.add_task(d3.dot(u,ey), name='v')
snapshots.add_task(np.sqrt(d3.dot(u,ex)**2 + d3.dot(u,ey)**2), name='vortex')

#-----------------------------------------------------------------------------------------------------------------

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