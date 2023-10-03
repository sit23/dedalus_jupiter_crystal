"""

mpiexec -n 4 python3 ./original_files/sw_cart.py &&
mpiexec -n 4 python3 ./original_files/plot_snapshots.py ./original_files/snapshots/*.h5 --output ./original_files/frames &&
ffmpeg -r 20 -i ./original_files/frames/write_%06d.png ./original_files/sw_cart.mp4

"""

# Import required libraries
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# python debugger
import pdb

# Parameters
#------------

# Simulation units
meter = 1 / 6.37122e6               ## Radius of Earth
hour = 1
second = hour / 3600

# Numerical Parameters
Lx, Lz = 1, 1                       # Domain size
Nx, Nz = 128, 128                   # Number of grid points
dealias = 3/2
stop_sim_time = 20                  # t_end
timestepper = d3.RK222              # Runge-kutta (2,2,2) scheme
max_timestep = 1e-2
dtype = np.float64

# Earth-like parameters
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter


#------------------------------------------------------------------------------------

#------------------------
# DEDALUS CONFIGURATIONS
#------------------------

# Bases
coords = d3.CartesianCoordinates('x', 'y')                  # set up coordinates
dist = d3.Distributor(coords, dtype=dtype)                                                  
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)


# Fields both functions of x,y
# vary in x,y dimensions (bases)
h = dist.Field(name='h', bases=(xbasis,ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))


# Substitutions
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)                # unit vectors


#------------------------------------------------------------------------------------

#--------------------
# INITIAL CONDITIONS
#--------------------

# Set up basic operators
#------------------------

# Horizontal cross product
zcross = lambda A: d3.skew(A)

# Set colatitude and convert theta to cartesian coordinates
coscolat = dist.Field(name='coscolat', bases=(xbasis, ybasis))
coscolat['g'] = np.cos(np.sqrt((x)**2. + (y)**2) / R)


# Initial condition: balanced height
#------------------------------------
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*coscolat*zcross(u))")
problem.add_equation("integ(h) = 0")
solver = problem.build_solver()
solver.solve()


#------------------------------------------------------------------------------------

#-----------------------
# SHALLOW WATER PROBLEM
#-----------------------

# Problem and Solver
#--------------------

## LHS must be first order in temporal derivatives and linear
## RHS can be nonlinear and time-dependent terms but no temporal derivatives
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*coscolat*zcross(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Snapshots
#-----------

# Set up and save snapshots
snapshots = solver.evaluator.add_file_handler('./original_files/snapshots', sim_dt=0.1, max_writes=10)

# Add full fields - vorticity, height, PV and planetary vorticity
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')                                     
snapshots.add_task((2*Omega*coscolat-d3.div(d3.skew(u)))/h, name='PV')

#Add perturbation fields
snapshots.add_task(h, name='pheight') 

#------------------------------------------------------------------------------------

#----------------------
# NUMERICAL SIMULATION
#----------------------


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