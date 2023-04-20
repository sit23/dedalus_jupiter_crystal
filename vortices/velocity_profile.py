import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import pdb

# Parameters
#----------------------------------------------------------------------------------------------

# Steepness parameter
b1 = 1
b2 = 1.5
b3 = 2

# Radius of vortex (km)
rm = 1000

# maximum velocity of vortex
vm = 80


# Set up profile
#----------------------------------------------------------------------------------------------

# range of radius
r = np.arange(-0*rm, 10*rm)

# Velocity profile
v1 = vm * ( r / rm ) * np.exp( (1/b1) * ( 1 - (r/rm)**b1 ) )

v2 = vm * ( r / rm ) * np.exp( (1/b2) * ( 1 - (r/rm)**b2 ) )

v3 = vm * ( r / rm ) * np.exp( (1/b3) * ( 1 - (r/rm)**b3 ) )


# Create plot
#----------------------------------------------------------------------------------------------

plt.figure()
plt.plot(r, v1, label=r'$b=1$')
plt.plot(r, v2, label=r'$b=1.5$')
plt.plot(r, v3, label=r'$b=2$')
plt.legend()
plt.xlabel(r'$r$')
plt.ylabel(r'$v(r)$')
plt.title('Velocity profile for varying steepness')
plt.show()