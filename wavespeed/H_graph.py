import xarray as xar
import numpy as np
import matplotlib.pyplot as plt
import glob

# Call ordered files from directory
files = sorted(glob.glob('./wavespeed/Experiment 1/*.nc'))

# Is it possible to save g and H into the xarray??
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
g = 9.80616 * meter / second**2

# Create array for H
H_range = np.arange(0.1e4, 1.6e4, 0.1e4) * meter
# Start arrays for true c and numerical c
c_true = np.sqrt(g*H_range)
c_numeric = np.zeros(len(files))

#----------------------------------------------------------------------------

# set counter for loop
counter = 0

# start loop that calls each file in turn
for f in files:
    ds = xar.open_dataset(f)

    height_values = ds.pheight.sel(y=-0.5, x=0)

    # turning point function
    def turning_points(list):
        dx = np.diff(list)                              
        return np.where(dx[1:] * dx[:-1] < 0)           

    # First turning point index location
    first_tp = turning_points(height_values)[0][0]

    # Location of first maximum point
    dt = height_values.t.data[first_tp]
    dy = - ds.height.y.data[0]

    c_numeric[counter] = dy / dt

    counter += 1

#----------------------------------------------------------------------------

# PLOTS

# make an array of strings for the xticks
xx = np.arange(0.1, 1.6, step=0.1)

# create function for mapping
def convert_string(element):
    el = round(element, 1)
    return str(el)

# make a list out of the map function
xticks = list(map(convert_string, xx))


# plot for true vs numerical
#-----------------------------

plt.figure(figsize=(6,9))
plt.subplot(2, 1, 1)
plt.scatter(H_range, c_numeric, label="Numerical", color='darkturquoise')
plt.scatter(H_range, c_true, label="True", color='blueviolet')
plt.xticks(H_range, xticks)
plt.xlabel(r'$H (10^4)$')
plt.ylabel(r'$c$')
plt.legend(loc='upper left')
plt.title('True and numerical wavespeed against equilibrium height')

# difference plot
#------------------

difference = abs(c_numeric - c_true)

# plt.subplot(1, 2, 2)
# plt.plot(H_range, difference, color='firebrick')
# plt.ylim(bottom=0)
# plt.xticks(H_range, xticks)
# plt.xlabel(r'$H (10^4)$')
# plt.ylabel('Difference')
# plt.title('Difference plot of true vs numerical results')
# plt.savefig('H_exp1.pdf')


# Percentage error
#------------------

percentage_error = (difference / c_true) * 100

plt.subplot(2, 1, 2)
plt.plot(H_range, percentage_error, color='firebrick')
plt.xticks(H_range, xticks)
plt.xlabel(r'$H (10^4)$')
plt.ylabel('Percentage')
plt.title('Percentage Error Plot')
plt.tight_layout()
plt.savefig('./wavespeed/Perc_error_exp1.png')