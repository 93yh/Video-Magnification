import matplotlib.pyplot as plt
from video_processing import Video
from video_processing import Video_Magnification
import numpy as np


video_path = 'video_samples/vibration.mp4'

# set the video object
video = Video(video_path)
video.set_frames()

# start video magnification
video_magnification = Video_Magnification(video)
# pre-processing
video_magnification.standardize_frames()
video_magnification.create_video_from_frames('frames_gray.avi')
video_magnification.apply_filter()
video_magnification.create_video_from_frames('magnitude_spectrum.avi', video_magnification.magnitude_frames)
video_magnification.create_video_from_frames('frames_pos_filtering.avi')
video_magnification.create_time_series()
phase_zero, amplitude_mean = video_magnification.remove_background()
# dimension reduction
dimension_reduced_series, principal_components = video_magnification.apply_PCA()
# blind source separation
eigen_values, eigen_vectors = video_magnification.apply_BSS(principal_components)
# video reconstruction
W = eigen_vectors.T
mode_shapes = np.dot(principal_components.T, W)
modal_coordinates = np.matmul(W, dimension_reduced_series.T)
video_magnification.video_reconstruction(mode_shapes, modal_coordinates, phase_zero)

# plt.plot(modal_coordinates[1, :])
# plt.imshow(mode_shapes[0].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray')
# plt.clf()