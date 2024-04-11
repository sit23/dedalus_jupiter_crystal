"""

mpiexec -n 16 python3 intruder_phys_vortex_forcing.py &&
mpiexec -n 16 python3 plot_intruder.py ./snapshots/intruder_forced_1_snapshots/*.h5 --output ./frames/intruder_frames &&
ffmpeg -r 120 -i ./reproduce/intruder/intruder_frames/write_%06d.png ./reproduce/intruder/intruder_h1e-8.mp4


Stitching two mp4s together:
    - ffmpeg -i ./reproduce/intruder/intruder_h0.mp4 -i ./reproduce/intruder/intruder_h1e-10.mp4 -filter_complex hstack ./reproduce/intruder/h0_h1e-10.mp4

"""


import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
from mpi4py import MPI

import pdb

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the current process
num_cores = comm.Get_size()
# Parameters
#------------

# Simulation units
meter = 1 / 71.4e6
day = 1
hour = day / 24
second = hour / 3600

# Numerical Parameters
Lx, Lz = 0.7, 0.7
Nx, Nz = 512, 512
dealias = 3/2                   
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Length of simulation (days)
stop_sim_time = 1.0
printout = 0.1
 
# Planetary Configurations
R = 71.4e6 * meter           
Omega = 1.74e-4 / second            
nu = 1e2 * meter**2 / second / 32**2 
g = 24.79 * meter / second**2

# energy injected at rate epsilon on a Gaussian ring N(kf, kfw) 
epsilon = 0.001     
kf = 2 * 14 * 2*np.pi/Lx
kfw = 1.5 * 2*np.pi/Lx
seed = None     

exp_name = 'example_intruder_phys_forcing_vortex_16_cores_sync_global_av3'


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

# forcing
Fh = dist.Field(name="Fh", bases=(xbasis, ybasis))

# Initialize arrays for storm times, latitudes, and longitudes
storm_time = np.zeros(31)
storm_lat = np.zeros(31)
storm_lon = np.zeros(31)
storm_count = 0

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

# Initial condition: South pole vortices
#----------------------------------------

# South pole coordinates
south_lat = [90., 85., 85., 85., 85., 85., 75.]
south_long = [0., 0., 72., 144., 216., 288., 0.]

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


def vortex_forcing(model_time, phi0, storm_count, storm_time, storm_lat, storm_lon):

    dt_hg_physical_forcing = np.zeros_like(Fh['g'])

    storm_strength = 1.0

    storm_interval = 10000.0  # 0.5 * (10**therm_damp_time) / 100.0 was commented out in the original
    storm_length = 10000.0  # Same as above

    h_width = 2.0
    h_width_dist_units = np.deg2rad(h_width)*R

    local_sum_of_forcing = np.array([0.], dtype='float64')
    global_sum_of_forcing = np.array([0.], dtype='float64')

    if rank == 0:

        if model_time == 0.0:
            storm_count = 0
            storm_strength = 0.0
            storm_time[0] = storm_interval * 1.5

            # Randomly generate storm location
            storm_lon[0] = np.random.rand() * 360
            temp_rand = np.random.rand()
            # storm_lat[storm_count] = -(90. - 45. * np.arccos(2 * temp_rand - 1) / np.arctan(1.0))
            storm_lat[0] = 80 + 10.*(np.random.rand()-0.5)  

        elif (model_time - (storm_time[storm_count]-storm_length/2.)) >= storm_interval:

            # Update storm count
            storm_count = (storm_count + 1) % 31

            # Set up future storm time
            if storm_count == 0:
                storm_time[storm_count] = storm_time[30] + storm_interval
            else:
                storm_time[storm_count] = storm_time[storm_count - 1] + storm_interval

            # Randomly generate storm location
            storm_lon[storm_count] = np.random.rand() * 360
            temp_rand = np.random.rand()
            # storm_lat[storm_count] = -(90. - 45. * np.arccos(2 * temp_rand - 1) / np.arctan(1.0))
            storm_lat[storm_count] = 80 + 10.*(np.random.rand()-0.5)            

    try:
        storm_time = comm.bcast(storm_time, root=0)
        storm_lon = comm.bcast(storm_lon, root=0)    
        storm_lat = comm.bcast(storm_lat, root=0)    
        storm_count = comm.bcast(storm_count, root=0)     
    except Exception as e:
        print(f"MPI Exception: {e}")

    # Loop through potential storm times to apply effects
    for storm_count_i in range(31):
        time_delta = model_time - storm_time[storm_count_i]
        storm_active = -storm_length / 2 <= time_delta <= storm_length / 2
        if storm_active and storm_time[storm_count_i] != 0:
            tt = (time_delta ** 2) / storm_length ** 2
            storm_strength = 1.0 * phi0 / storm_length
            x_loc_storm,y_loc_storm = conversion(storm_lat[storm_count_i], storm_lon[storm_count_i])
            xx = (x - x_loc_storm)/ (h_width_dist_units)
            yy = (y - y_loc_storm)/ (h_width_dist_units)
            dd = xx ** 2 + yy ** 2
            dt_hg_physical_forcing += storm_strength * np.exp(-dd) * np.exp(-tt)

    local_sum_of_forcing = np.array([np.sum(dt_hg_physical_forcing)])

    comm.Allreduce([local_sum_of_forcing, MPI.DOUBLE], [global_sum_of_forcing, MPI.DOUBLE], op=MPI.SUM)

    dt_hg_physical_forcing -= global_sum_of_forcing[0]/(dt_hg_physical_forcing.size*num_cores)

    return dt_hg_physical_forcing, storm_count, storm_time, storm_lat, storm_lon

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
h['g'] += ( np.random.rand(h['g'].shape[0], h['g'].shape[1]) - 0.5 ) * 1e-8



#-----------------------------------------------------------------------------------------------------------------

# Problem and Solver
#--------------------

# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*coscolat*zcross(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u) + Fh")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Snapshots
#-----------

# Set up and save snapshots
output_folder = f'snapshots/{exp_name}'
output_command = f'mpiexec -n 4 python3 plot_intruder.py {output_folder}/*.h5 --output=./frames/{exp_name}'
# Analysis
snapshots = solver.evaluator.add_file_handler(output_folder, sim_dt=printout, max_writes=10)

# add potential vorticity field
snapshots.add_task((-d3.div(d3.skew(u)) + 2*Omega*coscolat) / phi, name='PV')
snapshots.add_task(h, name='height')

snapshots.add_task(Fh, name='height_forcing')


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
        # Set vorticity forcing field from normalized Gaussian random field rescaled by forcing rate, including factor for 1/2 in kinetic energy
        # epsilon * kf**2 = enstrophy injection rate
        Fh.change_scales(1.)
        Fh["g"], storm_count, storm_time, storm_lat, storm_lon = vortex_forcing(solver.sim_time/second, phi0, storm_count, storm_time, storm_lat, storm_lon)

        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
    if rank==0:
        print(f'Please now run the following code for output processing - {output_command}')

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()