from collections import deque

import numpy as np
import cv2

from feature_extraction import single_img_features


class SearchWindows:
    """
    Class intended to decrease false detection and make new detected rectangles align with the previously detected ones
    """
    def __init__(self, n):
        # number of windows to store
        self.n = n
        # last n windows "hot" windows stored
        self.recent_windows = deque([], maxlen=n)
        # current windows
        self.current_windows = None
        # combined windows for analysis
        self.combined_windows = []

    def combine_windows(self):
        """
        Combines recent detected windows to provide better filtering from false positives
        """
        windows = []
        for local_windows in self.recent_windows:
            windows += local_windows
        if len(windows) == 0:
            self.combined_windows = None
        else:
            self.combined_windows = windows

    def update_tracking(self, windows):
        """
        Updates class with new "hot" windows
        :param windows: hot windows coordinates
        """
        self.current_windows = windows
        self.recent_windows.appendleft(self.current_windows)
        self.combine_windows()


def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Detect cars at the image using search windows approach

    :param img: image
    :param windows: windows list
    :param clf: classifier
    :param scaler: scale function
    :param color_space: selected color space
    :param spatial_size: spacial size
    :param hist_bins: number of histogram bins
    :param orient: number of orientations
    :param pix_per_cell: number of pixels per cell
    :param cell_per_block: number of cells per block
    :param hog_channel: hog channel selected
    :param spatial_feat: true if spatial feature to be used
    :param hist_feat: true if histogram feature to be used
    :param hog_feat: true if hog feature to be used
    :return: windows with positive detections
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    """
    Add heat for each pixel in the detected boxes

    :param heatmap: heat image
    :param bbox_list: list of windows
    :return: heat image
    """
    # Iterate through list of bboxes
    if bbox_list:
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    """
    Apply threshold to filter out false positives

    :param heatmap: heat image
    :param threshold: threshold for positive detections
    :return: heat image
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    """
    Draws labeled boxes
    :param img: image to draw on
    :param labels: labels to draw
    :return:
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def search_different_windows(image, clf, scaler, color_space='HLS', spatial_size=(32, 32), hist_bins=32, orient=9,
                             pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Generates different size search windows to look for cars

    :param image: image
    :param clf: classifier
    :param scaler: scale
    :param color_space: selected color space
    :param spatial_size: spatial size
    :param hist_bins: histogram bin number
    :param orient: number of orientations
    :param pix_per_cell: number of pixels per cell
    :param cell_per_block: number of cells per block
    :param hog_channel: selected hog channel
    :param spatial_feat: true if spatial feature to be used
    :param hist_feat: true if histogram feature to be used
    :param hog_feat: true if hog feature to be used
    :return: hot windows
    """
    hot_windows = []
    overlap = 0.75
    width1 = 256
    width2 = 192
    width3 = 128
    width4 = 64

    x_start_stop = [[None, None], [None, None], [None, None], [None, None]]
    xy_window = [(width1, width1), (width2, width2), (width3, width3), (width4, width4)]
    xy_overlap = [(overlap, overlap), (overlap, overlap), (overlap, overlap), (overlap, overlap)]
    yi0, yi1, yi2, yi3 = 380, 380, 395, 405
    y_start_stop = [[yi0, yi0 + width1 / 2], [yi1, yi1 + width2 / 2], [yi2, yi2 + width3 / 2], [yi3, yi3 + width4 / 2]]

    for i in range(len(y_start_stop)):
        windows = slide_window(image, x_start_stop=x_start_stop[i], y_start_stop=y_start_stop[i],
                               xy_window=xy_window[i], xy_overlap=xy_overlap[i])

        hot_windows += search_windows(image, windows, clf, scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat)

    return hot_windows


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Use sliding windows approach to generate all possible search windows

    :param img: image
    :param x_start_stop: coordinates for x start stop positions
    :param y_start_stop: coordinates for y start stop positions
    :param xy_window: windows size
    :param xy_overlap: windows overlap
    :return: search windows list
    """
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws bounding boxes

    :param img: image
    :param bboxes: boxes coordinates
    :param color: color of the box
    :param thick: thickness of the lines
    :return: image with drawn boxes
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy