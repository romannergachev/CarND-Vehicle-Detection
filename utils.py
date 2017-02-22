import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from feature_extraction import get_hog_features


def save_image(image1, image2, save, gray=False, gray2=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if gray2:
        ax1.imshow(image1, cmap='gray')
    else:
        ax1.imshow(image1)

    ax1.set_title('Original Image', fontsize=50)
    if gray:
        ax2.imshow(image2, cmap='gray')
    else:
        ax2.imshow(image2)

    ax2.set_title('Modified Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save:
        plt.savefig(save)
        plt.close(f)


def color_hist_single_ch(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    ch_hist = np.histogram(img, bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = ch_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Return the individual histogram and bin_centers
    return ch_hist, bin_centers


def test_hist():
    image = cv2.imread('non-vehicles/train/extra27.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    xyz = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    color_spaces_names = ['rgb', 'hsv', 'hls', 'luv', 'lab', 'xyz', 'yuv', 'ycc']
    color_spaces = [image, hsv, hls, luv, lab, xyz, yuv, ycc]

    rows = len(color_spaces)

    fig, axis = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    for row, colorspace in enumerate(color_spaces):
        axis[row, 0].set_title(color_spaces_names[row])
        axis[row, 0].imshow(colorspace)
        axis[row, 0].axis('off')
        for ch in range(3):
            ch_hist, bincen = color_hist_single_ch(colorspace[:, :, ch], nbins=32, bins_range=(0, 256))
            axis[row, ch + 1].set_title(color_spaces_names[row][ch])
            axis[row, ch + 1].bar(bincen, ch_hist[0])
    plt.savefig('output_images/histogram_notcar2')
    plt.close(fig)


def plot3d(pixels, colors_rgb,
        axis_labels, axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    plt.savefig('output_images/color_car2')
    plt.close(fig)

    return ax  # return Axes3D object for further manipulation


def hog_examples_drawing(image, color_space='YCrCb', orient=12, pix_per_cell=16, cell_per_block=2):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    fig, axarr = plt.subplots(1, 4, figsize=(12,3))
    axarr[0].set_title('original')
    axarr[0].imshow(image)

    _, ch_0 = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True)
    axarr[1].set_title('Channel 0')
    axarr[1].imshow(ch_0, cmap='jet')

    _, ch_1 = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, vis=True)
    axarr[2].set_title('Channel 1')
    axarr[2].imshow(ch_1, cmap='jet')

    _, ch_2 = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, vis=True)
    axarr[3].set_title('Channel 2')
    axarr[3].imshow(ch_2, cmap='jet')

    plt.savefig('output_images/hog_notcar_YCrCb')
    plt.close(fig)

