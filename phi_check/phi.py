"""

I want to compare my results for phi and their analytical solution of phi

"""

import numpy as np
import scipy.special as sc
import dedalus.public as d3
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
 
# Planetary Configurations
R = 69.911e6 * meter           
Omega = 1.76e-4 / second 
g = 24.79 * meter / second**2
H = 5e4 * meter


#-----------------------------------------------------------------------------------------------------------------

b = 1.5

# Calculate max speed with Rossby Number
Ro = 0.2
f0 = 2 * Omega                        
rm = 1e6 * meter                            
vm = Ro * f0 * rm                                

# Calculate Burger Number -- Currently Bu ~ 10
phi0 = g*H
Bu = phi0 / (f0 * rm)**2 

print(Bu)
print(vm * second / meter)
print( (phi0 * second**2 / meter**2) / 1e6)

#-----------------------------------------------------------------------------------------------------------------

x = np.linspace(-0.5, 0.5, 101) 
y = np.linspace(-0.5, 0.5, 101) 

r = np.sqrt(x**2 + y**2)

gamma = sc.gammainc(2/b, (1/b)*(r/rm)**b)

phi = phi0 * (1 - (Ro/Bu) * np.exp(1/b) * b**(2/b -1) * gamma )

pdb.set_trace()