import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import imageio

from search_windows import search_different_windows, add_heat, apply_threshold, draw_labeled_bboxes, SearchWindows, draw_boxes
from feature_extraction import *


def test_on_image(name):
    """
    Test pipeline for image recognition
    :param name: filename
    """
    image = cv2.imread('test_images/' + name + '.jpg')

    draw_image = np.copy(image)
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
    print(draw_image)

    hot_windows, all_windows = search_different_windows(image, svc, scaler, color_space=color_space,
                                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                                        orient=orient, pix_per_cell=pix_per_cell,
                                                        cell_per_block=cell_per_block,
                                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                        hist_feat=hist_feat, hog_feat=hog_feat)

    all_boxes = draw_boxes(draw_image, all_windows)

    heat = np.zeros_like(draw_image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 3)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    all_boxes = cv2.cvtColor(all_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_images/' + name + '_boxes.png', all_boxes)


def detection_pipeline(image):
    """
    Test pipeline for video recognition
    :param image: image from the video
    :return: masked image
    """
    draw_image = np.copy(image)
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
    hot_windows, all_windows = search_different_windows(draw_image, svc, scaler, color_space=color_space,
                                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                                        orient=orient, pix_per_cell=pix_per_cell,
                                                        cell_per_block=cell_per_block,
                                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                        hist_feat=hist_feat, hog_feat=hog_feat)

    windows.update_tracking(hot_windows)
    heat = np.zeros_like(draw_image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, windows.combined_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 20)
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

# feature extraction parameters
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# extract features from the data sets
car_train_features, car_test_features, notcar_train_features, notcar_test_features = extract_features_from_datasets(color_space, spatial_size, hist_bins, orient,
                                                                                                                    pix_per_cell, cell_per_block, hog_channel,
                                                                                                                    spatial_feat, hist_feat, hog_feat)
# rescale the data
x_stacked = np.vstack((car_train_features, notcar_train_features, car_test_features, notcar_test_features)).astype(np.float64)
scaler = StandardScaler().fit(x_stacked)
scaled_x = scaler.transform(x_stacked)

# create labels vectors
y_tr = np.hstack((np.ones(len(car_train_features)), np.zeros(len(notcar_train_features))))
y_tst = np.hstack((np.ones(len(car_test_features)), np.zeros(len(notcar_test_features))))

# generate random state for shuffle
rand_state = np.random.randint(0, 100)
# shuffle training and testing data
x_train, y_train = shuffle(scaled_x[:(len(car_train_features) + len(notcar_train_features))], y_tr, random_state=rand_state)
x_test, y_test = shuffle(scaled_x[(len(car_train_features) + len(notcar_train_features)):], y_tst, random_state=rand_state)

print('Feature vector length:', len(x_train[0]))

# create linear SVC
svc = LinearSVC()
t = time.time()
# fit linear SVC
svc.fit(x_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

test_on_image('test4')
# test_on_image('video1')
# test_on_image('video2')
# test_on_image('video3')
# test_on_image('video4')
# test_on_image('video5')

# create windows tracking class instance
windows = SearchWindows(20)
# use detection pipeline on the video
imageio.plugins.ffmpeg.download()
white_output = 'project_video_annotated.mp4'
clip1 = VideoFileClip("project_video.mp4")
# white_clip = clip1.fl_image(detection_pipeline)
# white_clip.write_videofile(white_output, audio=False)
