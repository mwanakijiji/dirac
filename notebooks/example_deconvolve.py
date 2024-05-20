import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))

# true object ersatz: checkerboard
checker = np.zeros((100, 100))
subarray_size = 10
for i in range(0, checker.shape[0], 2*subarray_size):
    for j in range(0, checker.shape[1], 2*subarray_size):
        checker[i:i+subarray_size, j:j+subarray_size] = 1

# impulse response ersatz: 2D Gaussian
x, y = np.meshgrid(np.arange(100), np.arange(100))
center_x = 50
center_y = 50
width = 1
gaussian_array = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))

# define the true object and the PSF
star = checker
psf = gaussian_array

# add noise
'''
star += np.random.normal(scale = 0.1, size = np.shape(star))
psf += np.random.normal(scale = 0.1, size = np.shape(star))
'''

star_conv = convolve(star, psf)
star_deconv = deconvolve(star_conv, psf) # noiseless

#import ipdb; ipdb.set_trace()

f, axes = plt.subplots(2,2)
axes[0,0].imshow(star)
axes[0,1].imshow(psf)
axes[1,0].imshow(np.real(star_conv))
axes[1,1].imshow(np.real(star_deconv))
plt.show()