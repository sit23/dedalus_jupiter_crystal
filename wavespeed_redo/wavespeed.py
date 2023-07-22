"""
This code solves the nonlinear rotating shallow water equations on a Cartesian grid centred on the pole (0,0), and a colatitude description of the Coriolis term. 

The parameters are currently set to Earth, but could easily be changed to Jupiter.

Plans:

* Do simple test case of SW wave propagation speed - does it work? 
* Do a geostrophic adjustment test
* Check equations properly
* Adapt parameters to Jupiter as per their paper, and maybe think about implementing the boundary drag.
* Should shrink domain so that it's the size of their domain - our's is almost certainly too big at this stage.



mpiexec -n 4 python3 ./wavespeed_redo/wavespeed.py &&
mpiexec -n 4 python3 ./wavespeed_redo/ded_to_xarray.py


mpiexec -n 4 python3 ./wavespeed_redo/wavespeed.py &&
mpiexec -n 4 python3 ./wavespeed_redo/plot_wavespeed.py ./wavespeed_redo/wavespeed_snapshots/*.h5 --output ./wavespeed_redo/wavespeed_frames &&
ffmpeg -r 20 -i ./wavespeed_redo/wavespeed_frames/write_%06d.png ./wavespeed_redo/wavespeed20.mp4 

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pdb
import matplotlib.pyplot as plt

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Numerical Parameters
Lx, Lz = 2, 2                
Nx, Nz = 256, 256                
dealias = 3/2
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Simulation length
stop_sim_time = 20
printout = 0.1

R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e2 * meter**2 / second / 32**2 
g = 9.80616 * meter / second**2
H = 1e4 * meter


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

zcross = lambda A: d3.skew(A)

coscolat = dist.Field(name='coscolat', bases=(xbasis, ybasis))
coscolat['g'] = np.cos(np.sqrt((x)**2. + (y)**2) / R)


# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*coscolat*zcross(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Initial conditions

#Create gaussian disturbance in height at the centre of the domain
h['g'] = H*0.01*np.exp(-((x)**2 + y**2)*100.)

# Make corrections for taking H out of above
hh = H + h

# pdb.set_trace()


# Analysis
snapshots = solver.evaluator.add_file_handler('./wavespeed_redo/wavespeed_snapshots', sim_dt=printout, max_writes=10)


#Add perturbation height field
snapshots.add_task(h, name='pheight')                                                               # perturbation height


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