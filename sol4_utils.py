from scipy.signal import convolve2d
import imageio
from skimage.color import rgb2gray
from imageio import imread, imwrite
import numpy as np
from scipy import signal
import scipy.signal



GREY = 1
NORMALIZATION_FACTOR = 2
MIN_IM_DIM = 16






"""
this function gets as parameters image path and its representation and read the image at the specified representation"""
def read_image(filename, representation):
    im_float = imageio.imread(filename)
    type = im_float.dtype
    if type == int or type == np.uint8:
        im_float = im_float.astype(np.float64) / 255
    if representation == GREY:
        return rgb2gray(im_float)
    return im_float



def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()



def reduce_im(im, gaus_filter):
    blurred = gaus_blurr(im, gaus_filter)
    return blurred[1::2, 1::2]


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

def gaus_blurr(im, gaus_filter):
    blurred = scipy.signal.convolve2d(im, gaus_filter, "same")
    x =  scipy.signal.convolve2d(blurred, gaus_filter.T, "same")
    return x


def expand_im(im, gaus_filter):
    x, y = im.shape
    padded_im = np.zeros((x * 2, y * 2))
    padded_im[::2, ::2] = im
    return gaus_blurr(padded_im, gaus_filter) * 2


def generate_filter(filter_size):
    conv = np.array([[1, 1]])
    in_process = np.array([[1, 1]]) / NORMALIZATION_FACTOR
    for i in range(filter_size - 2):
        in_process = scipy.signal.convolve2d(in_process, conv) / NORMALIZATION_FACTOR
    return in_process



def build_gaussian_pyramid(im, max_levels, filter_size):
    min_dim = np.min(im.shape)
    gaus_filter = generate_filter(filter_size)
    cur_level = im
    pyr = [im]
    for i in range(max_levels - 1):
        if (min_dim / 2) < MIN_IM_DIM:
            break
        min_dim /= 2
        cur_level = reduce_im(cur_level, gaus_filter)
        pyr.append(cur_level)
    return pyr, gaus_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr, gaus_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplac_pyr = []
    for i in range(len(pyr) - 1):
        exp = expand_im(pyr[i + 1], gaus_filter)
        laplac_pyr.append(pyr[i] - expand_im(pyr[i + 1], gaus_filter))
    laplac_pyr.append(pyr[len(pyr) - 1])
    return laplac_pyr, gaus_filter


