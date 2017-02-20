import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label

from test import *

# Read in cars and notcars
vehicles_train = glob.glob('vehicles/train/*/*.png')
vehicles_test = glob.glob('vehicles/test/*/*.png')
not_vehicles_train = glob.glob('non-vehicles/train/*.png')
not_vehicles_test = glob.glob('non-vehicles/test/*.png')

### TODO: Tweak these parameters and see how the results change.
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# LUV, channel = 1
# YUV, channel = 2
orient = 12  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

car_train_features = extract_features(vehicles_train, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat)
car_test_features = extract_features(vehicles_test, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)
notcar_train_features = extract_features(not_vehicles_train, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)
notcar_test_features = extract_features(not_vehicles_test, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
rand_state = np.random.randint(0, 100)

x_stacked = np.vstack((car_train_features, notcar_train_features, car_test_features, notcar_test_features)).astype(np.float64)
scaler = StandardScaler().fit(x_stacked)
scaled_x = scaler.transform(x_stacked)

y_tr = np.hstack((np.ones(len(car_train_features)), np.zeros(len(notcar_train_features))))
x_train, y_train = shuffle(scaled_x[:(len(car_train_features) + len(notcar_train_features))], y_tr, random_state=rand_state)

y_tst = np.hstack((np.ones(len(car_test_features)), np.zeros(len(notcar_test_features))))
x_test, y_test = shuffle(scaled_x[(len(car_train_features) + len(notcar_train_features)):], y_tst, random_state=rand_state)

print('Feature vector length:', len(x_train[0]))

svc = LinearSVC()
t = time.time()
svc.fit(x_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

image = cv2.imread('test_images/test1.jpg')
draw_image = np.copy(image)

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[np.int(image.shape[0] / 2), None],
                       xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

cv2.imwrite('output_images/test1_new1.png', window_img)

heat = np.zeros_like(draw_image[:, :, 0]).astype(np.float)
# Add heat to each box in box list
heat = add_heat(heat, hot_windows)
# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)
# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)
# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

cv2.imwrite('output_images/test1_heat1.png', draw_img)
