import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import glob

stem = '../data/xeneth_frames/20231108/low_gain/'
#stem = '../data/xeneth_frames/20231108/high_gain/'

file_list_all = glob.glob(stem + '*.png')
dark_files = sorted([file for file in file_list_all if 'dark' in file])
light_source_on_files = sorted([file for file in file_list_all if 'light_source_on' in file])
light_source_off_files = sorted([file for file in file_list_all if 'no_source' in file])

# make dark
dark_cube = np.array([iio.imread(file) for file in dark_files])

# make light (source on)
light_source_on_cube = np.array([iio.imread(file) for file in light_source_on_files])

# make light (source off)
light_source_off_cube = np.array([iio.imread(file) for file in light_source_off_files])


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


