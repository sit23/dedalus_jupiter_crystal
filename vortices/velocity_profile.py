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
vm_north = 75
vm_south = 80


# Set up profile
#----------------------------------------------------------------------------------------------

# range of radius for each pole
r_north = np.arange(0, 5*rm_north)
r_south = np.arange(-5*rm_north, 0)


# Velocity profile
#------------------

# Vary steepness parameters to be looped through
bb = [1,2,3]

# Create empty lists to appended in loop
v_north = []
v_south = []

for b in bb:

    vn = vm_north * ( r_north / rm_north ) * np.exp( (1/b) * ( 1 - ( r_north/rm_north )**b ) )
    vs = vm_south * ( r_south / -rm_south ) * np.exp( (1/b) * ( 1 - ( r_south/-rm_south )**b ) )

    v_north.append(vn)
    v_south.append(vs)




# Create plot
#----------------------------------------------------------------------------------------------

plt.figure()

plt.plot(r_north, v_north[0], color='#1f77b4', label=r'$b=1$')
plt.plot(r_north, v_north[1], color='#ff7f0e', label=r'$b=1.5$')
plt.plot(r_north, v_north[2], color='#2ca02c', label=r'$b=2$')
plt.plot(r_south, v_south[0], color='#1f77b4')
plt.plot(r_south, v_south[1], color='#ff7f0e')
plt.plot(r_south, v_south[2], color='#2ca02c')

plt.legend()
plt.xlabel('Distance (km)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity profile for varying steepness')
plt.savefig('./vortices/velocity_profile.png')