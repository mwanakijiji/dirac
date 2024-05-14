import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import glob
from photutils.centroids import centroid_quadratic
import matplotlib.colors as colors

stem = '../data/xeneth_frames/20240409/'


dark_file = stem + 'darks/dark_060.png'
#light_file = stem + 'zero_position/light_060.png'
#light_file = stem + 'position_closer_25mm/light_060.png'
#light_file = stem + 'position_closer_50mm/light_060.png'
#light_file = stem + 'position_further_25mm/light_060.png'
light_file = stem + 'position_further_50mm/light_060.png'

# make dark
dark_median = np.array(iio.imread(dark_file))

# light
light_median = np.array(iio.imread(light_file))


plt.imshow(light_median - dark_median, norm=colors.LogNorm())
plt.show()




'''
# medians
dark_median = np.median(dark_cube, axis=0)
light_source_off_median = np.median(light_source_off_cube, axis=0)
light_source_on_median = np.median(light_source_on_cube, axis=0)

# dark subtract
light_source_off_median = light_source_off_median-dark_median
light_source_on_median = light_source_on_median-dark_median

plt.imshow(light_source_off_median, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(light_source_on_median, origin='lower')
plt.colorbar()
plt.show()

plt.plot(light_source_on_median[210,:])
plt.show()
'''
