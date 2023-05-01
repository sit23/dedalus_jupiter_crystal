import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import pdb

# Simulation units
meter = 1 / 6.37122e6
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

# Earth Parameters
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter


# Bases
coords = d3.CartesianCoordinates('x', 'y')                                                  
dist = d3.Distributor(coords, dtype=dtype)                                                  
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)


# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))
h = dist.Field(name='h', bases=(xbasis,ybasis))


# Substitutions
x, y = dist.local_grids(xbasis,ybasis)
ex, ey = coords.unit_vector_fields(dist)

zcross = lambda A: d3.skew(A)


# INITIAL CONDITIONS

# Initial conditions: balanced height
c = dist.Field(name='c')
problem = d3.LBVP([h,c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("integ(h) = 0")                      ## ISSUE CAUSER
solver = problem.build_solver()
solver.solve()