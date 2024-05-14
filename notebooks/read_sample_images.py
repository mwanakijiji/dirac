import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import glob
from photutils.centroids import centroid_quadratic

stem = '../data/xeneth_frames/20231108/low_gain/'
#stem = '../data/xeneth_frames/20231108/high_gain/'
#stem = '../data/xeneth_frames/20240409/'

'''
file_list_darks = glob.glob(stem + 'darks/*.png')
file_list_lights = glob.glob(stem + 'zero_position/*.png')
'''
file_list_all = glob.glob(stem + '*.png')
dark_files = sorted([file for file in file_list_all if 'dark' in file])
light_source_on_files = sorted([file for file in file_list_all if 'light_source_on' in file])
light_source_off_files = sorted([file for file in file_list_all if 'no_source' in file])

# make dark
dark_cube = np.array([iio.imread(file) for file in dark_files])
dark_median = np.median(dark_cube, axis=0)

# make light (source on)
light_source_on_cube = np.array([iio.imread(file) for file in light_source_on_files])

# make light (source off)
light_source_off_cube = np.array([iio.imread(file) for file in light_source_off_files])

for frame_num in range(0,len(light_source_on_cube)):
    print(frame_num)
    frame = light_source_on_cube[frame_num] - dark_median
    x1, y1 = centroid_quadratic(frame)
    plt.imshow(frame, origin='lower')
    plt.plot([x1],[y1], color='red', marker='o')
    plt.savefig('centroid_{:03d}.png'.format(frame_num))
    plt.clf()





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
