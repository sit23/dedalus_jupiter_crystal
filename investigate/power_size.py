"""

mpiexec -n 16 python3 ./investigate/power_size.py &&
mpiexec -n 16 python3 ./investigate/plot_investigate.py ./investigate/investigate_snapshots/*.h5 --output ./investigate/investigate_frames &&
ffmpeg -r 40 -i ./investigate/investigate_frames/write_%06d.png ./investigate/size_0p5e6.mp4


"""


import numpy as np
import dedalus.public as d3
import logging 
logger = logging.getLogger(__name__)

import pdb


# Parameters
#------------

# Simulation units
meter = 1 / 71.4e6
day = 1
hour = day / 24
second = hour / 3600

# Numerical Parameters
Lx, Lz = 0.7, 0.7               # x,y = (0,0) to be ~70 lat
Nx, Nz = 512, 512
dealias = 3/2                   
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Length of simulation (days)
stop_sim_time = 500
printout = 0.25
 
# Planetary Configurations
R = 71.4e6 * meter           
Omega = 1.74e-4 / second            
nu = 1e2 * meter**2 / second / 32**2 
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
Ro = 0.23


# Dependent variables
#---------------------

# Calculate max speed with Rossby Number
f0 = 2 * Omega                                       # Planetary vorticity
rm = 1e6 * meter                                     # Radius of vortex (km)
vm = Ro * f0 * rm                                    # Calculate speed with Ro

# Calculate deformation radius with Burger number
H = 5e4 * meter 
phi = g * (h + H) 

# Calculate Burger Number -- Currently Bu ~ 10
phi0 = g*H
Bu = phi0 / (f0 * rm)**2 

# Check phi0 dimensionalised
phi00 = phi0 * second**2 / meter**2
# pdb.set_trace()



# Initial condition: South pole vortices
#----------------------------------------

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
    u['g'][0] += - vm * ( r / rm ) * np.exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (y-yy[i]) / ( r + 1e-16 ) )
    u['g'][1] += vm * ( r / rm ) * np.exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (x-xx[i]) / ( r + 1e-16 ) )   



# Initial condition: Intruder settings
#--------------------------------------

## maybe play around with steepness parameter too?

# intruder start location
lat_int = 75
long_int = 0

# intruder size
rm_int= 0.5e6 * meter

# intruder velocity
intruder_velocity = 80.04
vm_int= intruder_velocity * meter / second 


# convert parameters to cartesian like above
xx_int, yy_int = conversion(lat_int, long_int)
r_int = np.sqrt( (x-xx_int)**2 + (y-yy_int)**2 )

# Overide u,v components in velocity field
u['g'][0] += - vm_int * ( r_int / rm_int ) * np.exp( (1/b) * ( 1 - ( r_int / rm_int )**b ) ) * ( (y-yy_int) / ( r_int + 1e-16 ) )
u['g'][1] += vm_int * ( r_int / rm_int ) * np.exp( (1/b) * ( 1 - ( r_int / rm_int )**b ) ) * ( (x-xx_int) / ( r_int + 1e-16 ) )  

                        


# Initial condition: height
#---------------------------
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*coscolat*zcross(u))")
problem.add_equation("integ(h) = 0")
solver = problem.build_solver()
solver.solve()


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
snapshots = solver.evaluator.add_file_handler('./investigate/investigate_snapshots', sim_dt=printout, max_writes=10)

# add potential vorticity field
snapshots.add_task((-d3.div(d3.skew(u)) + 2*Omega*coscolat) / phi, name='PV')


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