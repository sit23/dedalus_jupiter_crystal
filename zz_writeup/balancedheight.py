import xarray as xar
import matplotlib.pyplot as plt

# Import datasets and choose location to graph
cyclone_balanced = xar.open_dataset('./balanced/cyclone_balanced.nc')
cyclone_not = xar.open_dataset('./balanced/cyclone_not.nc')

pheight_bal = cyclone_balanced.pheight[:, 300, 300]
pheight_not = cyclone_not.pheight[:, 300, 300]

# Plot figure and save to PNG file
plt.figure()
plt.title('Balanced height vs non-balanced height')
plt.plot(pheight_not.t, pheight_not.data, label='not balanced', color='firebrick')
plt.plot(pheight_bal.t, pheight_bal.data, label='balanced', color='lime')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('perturbation height (m)')
plt.savefig('./balanced/bal_height.png')
