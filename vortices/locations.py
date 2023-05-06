import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import pdb

# Simulation units
meter = 1 / 69.911e6

# Numerical Parameters
Lx, Lz = 1, 1

# Radius of vortex (km)
rm = 1e6 * meter

# Radius of Jupiter
R = 69.911e6 * meter 

#--------------------------------------------------------------------------------------------

# Longitude and latitude conversion
#------------------------------------

# South pole coordinates
south_lat = [88.6, 83.7, 84.3, 85.0, 84.1, 83.2]
south_long = [211.3, 157.1, 94.3, 13.4, 298.8, 229.7]

# North pole coordinates
north_lat =[89.6, 82.9, 83.8, 82.0, 83.2, 82.9, 83.2, 82.3, 83.5]
north_long =[230.4, 1.4, 50.7, 95.3, 137.6, 183.4, 227.6, 269.9, 314.8]


# Conversion
#------------

# Define conversion function
def conversion(lat, lon):

    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)

    return x, y


# Create list of cartesian coordinates
coords_south = []

for i in range(len(south_lat)):

    (x, y) = conversion(south_lat[i], south_long[i])
    coords_south.append((x,y))


#--------------------------------------------------------------------------------------------


# PLOTS

plt.figure(figsize=(5,5))
plt.xlim(-Lx/2, Lx/2)
plt.ylim(-Lz/2, Lz/2)

#Circles
for i in range(len(coords_south)):

    circle = plt.Circle( coords_south[i], radius=rm, color='darkred')
    plt.gca().add_patch(circle)

plt.show()