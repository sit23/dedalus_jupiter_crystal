"""

mpiexec -n 16 python3 intruder_phys_vortex_forcing_sphere.py &&
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
import ded3_xarray as dedxar
import json
import os

import pdb

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the current process
num_cores = comm.Get_size()
# Parameters
#------------

# Simulation units
meter = 1 / 7.14e7
day = 1
hour = day / 24
second = hour / 3600

# Numerical Parameters
Nphi = 512
Ntheta = 256
dealias = 3/2                   
timestepper = d3.RK222
max_timestep = 5e-3
dtype = np.float64

# Length of simulation (days)
stop_sim_time = 5000.0
printout = 100.0
 
# Planetary Configurations
R = 7.14e7 * meter           
Omega = 1.74e-4 / second            
nu = 1e16 * meter**4 / second #used to be about 1e-11, now 1e-25 (dimensions were wrong)
g = 24.79 * meter / second**2

phi0 = 6e4 *meter**2./second**2.

#parameter for radiative damping
inv_tau_rad = 0.0 #have made it the inverse of tau_rad so that tau_rad = infinity is easily done by setting inv_tau_rad = 0.0

#forcing parameters

showman_s0 = 0.333 #m^2/s^3
storm_interval_dim = 1.25e4 #seconds  
storm_length_dim = 100000.0   #seconds

h_width = 0.7 #degrees



showman_s0_nondim = showman_s0*meter**2./second**3
showman_s0_nondim_height_units = showman_s0_nondim/g


exp_name = 'showman_2007_A1_mk10'
output_folder = f'snapshots/{exp_name}'

params_to_store = ['meter', 'second', 'Nphi', 'Ntheta', 'dealias', 'max_timestep', 'stop_sim_time', 'printout', 'R', 'Omega', 'nu', 'g', 'inv_tau_rad', 'showman_s0', 'storm_interval_dim', 'storm_length_dim', 'h_width', 'exp_name', 'output_folder', 'phi0']

all_locals = locals()
    # Selectively create a dictionary of variables you are interested in
params = {key: all_locals[key] for key in all_locals if key in params_to_store}

if rank==0:
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    # Write dictionary to a file in JSON format
    with open(f'{output_folder}/sim_variables.json', 'w') as f:
        json.dump(params, f, indent=4)

#-----------------------------------------------------------------------------------------------------------------

# Dedalus set ups
#-----------------

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)


# Fields both functions of x,y
h = dist.Field(name='h', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)

ones_arr = dist.Field(name='ones', bases=basis)

ones_arr['g'] = 1.

# e_phi = dist.VectorField(coords, bases=basis) #(where coords and basis are as defined in the example). 
# e_phi['g'][0] = 1. 

# e_theta = dist.VectorField(coords, bases=basis) #(where coords and basis are as defined in the example). 
# e_theta['g'][1] = 1. 

# ephi, etheta = coords.unit_vector_fields(dist)

# Substitutions
lon, theta = dist.local_grids(basis)
#adding zero times lon and theta so that we get a 256x128 array of lat and lon
lat = np.pi / 2 - theta + 0*lon
lon = lon + 0.*theta

# forcing
Fh = dist.Field(name="Fh", bases=basis)

# Initialize arrays for storm times, latitudes, and longitudes
storm_time = np.zeros(31)
storm_lat = np.zeros(31)
storm_lon = np.zeros(31)
storm_count = 0

# Set up basic operators
zcross = lambda A: d3.MulCosine(d3.skew(A))

#-----------------------------------------------------------------------------------------------------------------

# INITIAL CONDITIONS for vortex crystal

# Independent variables
#-----------------------

# Steepness parameter
b = 1.5

# Rossby Number
Ro = 100.

add_vortex_crystal = False

# Dependent variables
#---------------------

# Calculate max speed with Rossby Number
f0 = 2 * Omega                                       # Planetary vorticity
rm = 1e6 * meter                                     # Radius of vortex (km)
vm = Ro * f0 * rm                                    # Calculate speed with Ro

rm_ang_rad = rm /R

# Calculate deformation radius with Burger number
# phi0 = Ro*4.*(((Omega*R))**2.) 

H = phi0/g
phi = g * (h + H) 

# Calculate Burger Number -- Currently Bu ~ 10
# phi0 = g*H
# Bu = phi0 / (f0 * rm)**2 

# Ek = nu/(2.*Omega)

# Check phi0 dimensionalised
phi00 = phi0 * second**2 / meter**2
if rank==0:
    print(f'non dim value of nu = {nu}')
    print(f'model units phi0={phi0}')
    print(f'physical units phi00={phi00}')

# Initial condition: South pole vortices
#----------------------------------------


# Convert longitude and latitude inputs into x,y coordinates
def conversion(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return lon, lat

if add_vortex_crystal:
    # South pole coordinates
    south_lat = [90., 85., 85., 85., 85., 85.]#, 75.]
    # south_lat = [90., 45., 45., 45., 45., 45., ]
    south_long = [0., 0., 72., 144., 216., 288.]
    # south_lat = [45.]
    # south_long = [270.]

    for i in range(len(south_lat)):

        # xx = (deg_lon(i) - (storm_lon(storm_count_i)+mm*360.))/(h_width/cos_lat(j))
        # yy = (deg_lat(j) - storm_lat(storm_count_i))/h_width

        # x_loc_storm,y_loc_storm = conversion(storm_lat[storm_count_i], storm_lon[storm_count_i])
        # 
        # 
        # dd = xx ** 2 + yy ** 2
        # dt_hg_physical_forcing += storm_strength * np.exp(-dd) * np.exp(-tt)


        south_lon_rad, south_lat_rad = conversion(south_lat[i], south_long[i])
        # r = np.sqrt( (x-xx[i])**2 + (y-yy[i])**2 )
        xx = (lon - south_lon_rad)/ (1./np.cos(lat))
        xx_m2pi = (lon - south_lon_rad-2.*np.pi)/ (1./np.cos(lat))
        xx_p2pi = (lon - south_lon_rad+2.*np.pi)/ (1./np.cos(lat))

        xx_min = np.zeros_like(xx) + np.nan
        where_xx_min = np.where(np.logical_and(np.abs(xx)<=np.abs(xx_m2pi), np.abs(xx)<=np.abs(xx_p2pi)))
        where_xx_m2pi_min = np.where(np.logical_and(np.abs(xx_m2pi)<=np.abs(xx), np.abs(xx_m2pi)<=np.abs(xx_p2pi)))
        where_xx_p2pi_min = np.where(np.logical_and(np.abs(xx_p2pi)<=np.abs(xx), np.abs(xx_p2pi)<=np.abs(xx_m2pi)))

        xx_min[where_xx_min] = xx[where_xx_min]
        xx_min[where_xx_m2pi_min] = xx_m2pi[where_xx_m2pi_min]    
        xx_min[where_xx_p2pi_min] = xx_p2pi[where_xx_p2pi_min]    

        assert(not np.any(np.isnan(xx_min)))

        xx_min_sqd = np.minimum(xx_p2pi**2., xx_m2pi**2.)
        xx_min_sqd = np.minimum(xx_min_sqd, xx**2.)
        yy = (lat - south_lat_rad) / (1.)

        if south_lat[i]==90. or south_lat[i] == -90.:
            r = np.abs(yy)
            xx_min = np.zeros_like(xx_min)
        else:
            r = np.sqrt(xx_min_sqd + yy**2.)

        u['g'][0] -= vm * ( r / rm_ang_rad ) * np.exp( (1/b) * ( 1 - ( r / rm_ang_rad )**b ) ) * ( (yy) / ( r + 1e-16 ) )
        u['g'][1] -= vm * ( r / rm_ang_rad ) * np.exp( (1/b) * ( 1 - ( r / rm_ang_rad )**b ) ) * ( (xx_min) / ( r + 1e-16 ) ) 


def vortex_forcing(model_time, H, storm_count, storm_time, storm_lat, storm_lon, storm_length, storm_interval):

    dt_hg_physical_forcing = np.zeros_like(Fh['g'])

    storm_strength = 1.0

    h_width_rad = np.deg2rad(h_width)

    local_sum_of_forcing = np.array([0.], dtype='float64')
    global_sum_of_forcing = np.array([0.], dtype='float64')

    local_sum_of_grid = np.array([0.], dtype='float64')
    global_sum_of_grid = np.array([0.], dtype='float64')

    if rank == 0:

        if model_time == 0.0:
            storm_count = 0
            storm_strength = 0.0
            storm_time[0] = storm_interval * 1.5

            # Randomly generate storm location
            storm_lon[0] = np.random.rand() * 360.
            temp_rand = np.random.rand()
            storm_lat[0] = -(90. - 45. * np.arccos(2 * temp_rand - 1) / np.arctan(1.0))

        elif (model_time - (storm_time[storm_count]-storm_length/2.)) >= storm_interval:

            # Update storm count
            storm_count = (storm_count + 1) % 31

            # Set up future storm time
            if storm_count == 0:
                storm_time[storm_count] = storm_time[30] + storm_interval
            else:
                storm_time[storm_count] = storm_time[storm_count - 1] + storm_interval

            # Randomly generate storm location
            storm_lon[storm_count] = np.random.rand() * 360.
            temp_rand = np.random.rand()
            storm_lat[storm_count] = -(90. - 45. * np.arccos(2 * temp_rand - 1) / np.arctan(1.0))

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
            storm_strength =  showman_s0_nondim_height_units #must be in non-dim units
            south_lon_rad, south_lat_rad = conversion(storm_lat[storm_count_i], storm_lon[storm_count_i])

            xx      = (lon - south_lon_rad         )/ (h_width_rad/np.cos(lat))
            xx_m2pi = (lon - south_lon_rad-2.*np.pi)/ (h_width_rad/np.cos(lat))
            xx_p2pi = (lon - south_lon_rad+2.*np.pi)/ (h_width_rad/np.cos(lat))

            xx_min_sqd = np.minimum(xx_p2pi**2., xx_m2pi**2.)
            xx_min_sqd = np.minimum(xx_min_sqd, xx**2.)

            yy = (lat - south_lat_rad) / (h_width_rad)         
               
            dd = xx_min_sqd + yy ** 2
            dt_hg_physical_forcing += storm_strength * np.exp(-dd) * np.exp(-tt)

    local_sum_of_forcing = np.array([np.sum(dt_hg_physical_forcing*np.cos(lat))])
    local_sum_of_grid    = np.array([np.sum(np.cos(lat))])

    comm.Allreduce([local_sum_of_forcing, MPI.DOUBLE], [global_sum_of_forcing, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce([local_sum_of_grid,    MPI.DOUBLE], [global_sum_of_grid,    MPI.DOUBLE], op=MPI.SUM)

    dt_hg_physical_forcing -= global_sum_of_forcing[0]/global_sum_of_grid[0]

    return dt_hg_physical_forcing, storm_count, storm_time, storm_lat, storm_lon

# Initial condition: height
#---------------------------
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h) = 0")
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
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  = - u@grad(u) - 2*Omega*zcross(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) + h*inv_tau_rad = - div(h*u) + Fh")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time 


# Snapshots
#-----------

# Set up and save snapshots
# output_command = f'mpiexec -n 4 python3 plot_sphere_phys_forcing.py {output_folder}/*.h5 --output=./frames/{exp_name}'
# Analysis
snapshots = solver.evaluator.add_file_handler(output_folder, sim_dt=printout, max_writes=10)

# add potential vorticity field
# snapshots.add_task((-d3.div(d3.skew(u)) + 2*Omega*coscolat) / phi, name='PV')
snapshots.add_task(h/meter, name='height')
snapshots.add_task((h+H)/meter, name='total_height')
snapshots.add_task(-d3.div(d3.skew(u))*second, name='vorticity')
snapshots.add_task(Fh*second/meter, name='height_forcing')
snapshots.add_task(u*second/meter, name='u')
snapshots.add_task(((2*Omega*d3.MulCosine(ones_arr)-d3.div(d3.skew(u)))/(h+H))*second*meter, name='PV')
# snapshots.add_task(0.5*u@u, name='e_kin')
# snapshots.add_task(0.5*h**2., name='APE')


#-----------------------------------------------------------------------------------------------------------------

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        # Set vorticity forcing field from normalized Gaussian random field rescaled by forcing rate, including factor for 1/2 in kinetic energy
        # epsilon * kf**2 = enstrophy injection rate
        Fh.change_scales(1.)
        Fh["g"], storm_count, storm_time, storm_lat, storm_lon = vortex_forcing(solver.sim_time/second, H, storm_count, storm_time, storm_lat, storm_lon, storm_length_dim, storm_interval_dim)

        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if rank==0:
        # print(f'Please now run the following code for output processing - {output_command}')
        dedxar.convert_to_netcdf(exp_name, force_recalculate=True)

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()