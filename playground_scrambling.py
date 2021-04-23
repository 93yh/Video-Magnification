from video_processing import Video
from video_processing import Video_Magnification
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


video_path = 'video_samples/vibration2.avi'
number_components = 16
components_order = np.arange(number_components)
sources_order = np.arange(number_components)
modal_coordinates_order = np.array([8, 9, 2, 3, 11, 12])
# modal_coordinates_order = np.array([0, 1, 4, 5, 6, 7])

# set the video object
video = Video(video_path)

# Start video magnification
video_magnification = Video_Magnification(video)
video_magnification.create_video_from_frames("frames_gray", frames=video.gray_frames)

# Create time series
time_serie = video_magnification.create_time_series()

# Scrambling time serie
encryption_key = video_magnification.scramble()

# Remove background
video_magnification.remove_background()

# Hibert Transform
real_time_serie, imag_time_serie = video_magnification.apply_hilbert_transform()

# dimension reduction in the real and imaginary time series
eigen_vectors, eigen_values, components = video_magnification.dimension_reduction()

# blind source separation
mixture_matrix, sources = video_magnification.extract_sources(number_components)

# create mode shapes and modal coordinates
mode_shapes, modal_coordinates = video_magnification.create_mode_shapes_and_modal_coordinates(number_components,
                                                                                              modal_coordinates_order)
# write in a file to use the same variable on matlab for debugging purposes
mdic = {"a": components, "label": "experiment"}
scipy.io.savemat('components.mat', mdic)

# vizualize principal components
video_magnification.visualize_components_or_sources('components', components_order)

# visualize sources
video_magnification.visualize_components_or_sources('sources', sources_order)

# visualize mode shapes without scrambling
video_magnification.visualize_mode_shapes_and_modal_coordinates(modal_coordinates_order, do_unscramble=True)

# visualize modal coordinates and mode shapes
video_magnification.visualize_mode_shapes_and_modal_coordinates(modal_coordinates_order)

# visualize only modal coordinates
video_magnification.visualize_components_or_sources('modal coordinates', np.arange(len(modal_coordinates_order)))

# video reconstruction
frames_0, frames_1, frames_2, frames_3 = video_magnification.video_reconstruction()
video_magnification.create_video_from_frames("mode0_U", frames=frames_0)
video_magnification.create_video_from_frames("mode1_U", frames=frames_1)
video_magnification.create_video_from_frames("mode2_U", frames=frames_2)
video_magnification.create_video_from_frames("mode3_U", frames=frames_3)

# video reconstruction
frames_0, frames_1, frames_2, frames_3 = video_magnification.video_reconstruction(do_unscramble=True)
video_magnification.create_video_from_frames("mode0", frames=frames_0)
video_magnification.create_video_from_frames("mode1", frames=frames_1)
video_magnification.create_video_from_frames("mode2", frames=frames_2)
video_magnification.create_video_from_frames("mode3", frames=frames_3)

# Calculate error
error, norm = video_magnification.calculate_error()
