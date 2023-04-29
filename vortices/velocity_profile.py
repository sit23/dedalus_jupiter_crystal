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

# Radius of vortices at each pole (km)
rm_north = 1000
rm_south = 1200

# maximum velocity of vortex (m/s)
vm = 80


# Set up profile
#----------------------------------------------------------------------------------------------

# range of radius for each pole
r_north = np.arange(0, 6*rm_north)
r_south = np.arange(-6*rm_north, 0)


# Velocity profile
v1 = vm * ( r_north / rm_north ) * np.exp( (1/b1) * ( 1 - ( r_north/rm_north )**b1 ) )
v2 = vm * ( r_north / rm_north ) * np.exp( (1/b2) * ( 1 - ( r_north/rm_north )**b2 ) )
v3 = vm * ( r_north / rm_north ) * np.exp( (1/b3) * ( 1 - ( r_north/rm_north )**b3 ) )

v1_south = vm * ( r_south / -rm_south ) * np.exp( (1/b1) * ( 1 - ( r_south/-rm_south )**b1 ) )
v2_south = vm * ( r_south / -rm_south ) * np.exp( (1/b2) * ( 1 - ( r_south/-rm_south )**b2 ) )
v3_south = vm * ( r_south / -rm_south ) * np.exp( (1/b3) * ( 1 - ( r_south/-rm_south )**b3 ) )


# Create plot
#----------------------------------------------------------------------------------------------

plt.figure()
plt.plot(r_north, v1, color='#1f77b4', label=r'$b=1$')
plt.plot(r_north, v2, color='#ff7f0e', label=r'$b=1.5$')
plt.plot(r_north, v3, color='#2ca02c', label=r'$b=2$')
plt.plot(r_south, v1_south, color='#1f77b4')
plt.plot(r_south, v2_south, color='#ff7f0e')
plt.plot(r_south, v3_south, color='#2ca02c')
plt.legend()
plt.xlabel('Distance (km)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity profile for varying steepness')
plt.show()