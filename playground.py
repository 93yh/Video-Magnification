from video_processing import Video
from video_processing import Video_Magnification
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


video_path = 'video_samples/vibration2.avi'
number_components = 16
components_order = np.arange(number_components)
sources_order = np.arange(number_components)
# modal_coordinates_order = np.array([8, 9, 2, 3, 11, 12])
# modal_coordinates_order = np.array([0, 1, 2, 3, 6, 7])

# set the video object
video = Video(video_path)

# Start video magnification
magnification = Video_Magnification(video)
magnification.create_video_from_frames("frames_gray", frames=video.gray_frames)

# Create time series
time_serie = magnification.create_time_series()

# Remove background
magnification.remove_background()

# Hibert Transform
real_time_serie, imag_time_serie = magnification.apply_hilbert_transform()

# dimension reduction in the real and imaginary time series
eigen_vectors, eigen_values, components = magnification.dimension_reduction()

# vizualize principal components
magnification.visualize_components_or_sources('components', components_order)

# blind source separation
mixture_matrix, sources = magnification.extract_sources(number_components)

# visualize sources
magnification.visualize_components_or_sources('sources', sources_order)

# create mode shapes and modal coordinates
mode_shapes, modal_coordinates = magnification.create_mode_shapes_and_modal_coordinates(number_components,
                                                                                              modal_coordinates_order)
                                                                                              
# visualize modal coordinates and mode shapes
magnification.visualize_mode_shapes_and_modal_coordinates(modal_coordinates_order)

# visualize only modal coordinates
magnification.visualize_components_or_sources('modal coordinates', np.arange(len(modal_coordinates_order)))

# video reconstruction
frames_0, frames_1, frames_2, frames_3 = magnification.video_reconstruction()
magnification.create_video_from_frames("mode0", frames=frames_0)
magnification.create_video_from_frames("mode1", frames=frames_1)
magnification.create_video_from_frames("mode2", frames=frames_2)
magnification.create_video_from_frames("mode3", frames=frames_3)

# Calculate error
error, norm = magnification.calculate_error()

# write in a file to use the same variable on matlab for debugging purposes
mdic = {"a": sources, "label": "experiment"}
scipy.io.savemat('sources.mat', mdic)
mdic = {"a": mixture_matrix, "label": "experiment"}
scipy.io.savemat('mixture.mat', mdic)
mdic = {"a": mode_shapes, "label": "experiment"}
scipy.io.savemat('shapes.mat', mdic)
mdic = {"a": modal_coordinates, "label": "experiment"}
scipy.io.savemat('coordinates.mat', mdic)
