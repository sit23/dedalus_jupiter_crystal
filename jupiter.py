"""
This code solves the nonlinear rotating shallow water equations on a Cartesian grid centred on the pole (0,0), and a colatitude description of the Coriolis term. 

The parameters are currently set to Earth, but could easily be changed to Jupiter.

Plans:

* Do simple test case of SW wave propagation speed - does it work? 
* Do a geostrophic adjustment test
* Check equations properly
* Adapt parameters to Jupiter as per their paper, and maybe think about implementing the boundary drag.
* Should shrink domain so that it's the size of their domain - our's is almost certainly too big at this stage.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 sw_cart.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
    $ mpiexec -n 4 python3 ded_to_xarray.py
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pdb

# Simulation units
#------------------
meter = 1 / 69.911e6            ## Radius of Jupiter (Use polar radius? 66.8)
hour = 1
second = hour / 3600

# Numerical Parameters
#----------------------
Lx, Lz = 1, 1
Nx, Nz = 512, 512
dealias = 3/2                   
stop_sim_time = 20
timestepper = d3.RK222          ## Why this time stepper?
max_timestep = 1e-2
dtype = np.float64

# Planetary Configurations
#---------------------------
R = 69.911e6 * meter                    ## Radius of Jupiter
Omega = 1.76e-4 / second                ## Rotation of Jupiter
nu = 1e5 * meter**2 / second / 32**2    # Hyperdiffusion matched at ell=32
g = 24.79 * meter / second**2           ## Jupiter gravity
H = 5e4 * meter                         ## Jupiter cloud depth

#--------------------------------------------------------------------------------------------

# Vortex Additions
#------------------
f = 2 * Omega               # coriolis
L = 1e6 * meter             # vortex radius
Lp = 8.7e6 * meter          # distance from pole to vortices
u = 80 * meter / second     # velocity

c = np.sqrt(g*H)
Ld = c / f


# Nondimensional parameters
#---------------------------
Ro = u / (f * L)
Bu = Ld**2 / L**2

pdb.set_trace()


#--------------------------------------------------------------------------------------------

# Bases
#-------
coords = d3.CartesianCoordinates('x', 'y')                                                  # set up coordinates
dist = d3.Distributor(coords, dtype=dtype)                                                  
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields both functions of x,y
h = dist.Field(name='h', bases=(xbasis,ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))


# Substitutions
#---------------
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

#--------------------------------------------------------------------------------------------

# My additions




#--------------------------------------------------------------------------------------------

# Definitions
#-------------
zcross = lambda A: d3.skew(A) # 90deg rotation anticlockwise (positive)

coscolat = dist.Field(name='coscolat', bases=(xbasis, ybasis))
coscolat['g'] = np.cos(np.sqrt((x)**2. + (y)**2) / R)                       # theta?

# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*coscolat*zcross(u)")
# problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u)") #               # Simulate no rotation
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Initial conditions

#Create gaussian disturbance in height at the centre of the domain
h['g'] = H*0.01*np.exp(-((x)**2 + y**2)*100.)

# Make corrections for taking H out of above
hh = H + h


# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)

#Add full fields - vorticity, height, PV and planetary vorticity
snapshots.add_task(hh, name='height') 
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')                                           # Why negative? - write out skew and derive
snapshots.add_task((2*Omega*coscolat-d3.div(d3.skew(u)))/hh, name='PV')
snapshots.add_task((2*Omega*coscolat), name='plvort')                           # coriolis term


#Add perturbation fields - i.e. full field minus initial values
snapshots.add_task(h, name='pheight')                                                               # perturbation height
snapshots.add_task((2*Omega*coscolat-d3.div(d3.skew(u)))/hh - (2*Omega*coscolat)/H, name='pPV')     # perturbation PV - hh or h?

#--------------------------------------------------------------------------------------------

# Add fields
#------------

# theta
theta = dist.Field(name='theta', bases=(xbasis, ybasis))
theta['g'] = Lp / R
# snapshots.add_task( Lp / R, name='theta')

snapshots.add_task( 2*Omega*theta / R, name='beta')

pdb.set_trace()


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