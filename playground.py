from video_processing import Video
from video_processing import Video_Magnification
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


video_path = 'video_samples/vibration.avi'
number_components = 8
components_order = np.arange(number_components)
sources_order = np.arange(number_components)
# modal_coordinates_order = np.array([8, 9, 2, 3, 11, 12])
modal_coordinates_order = np.array([0, 1, 2, 3, 6, 7])
# modal_coordinates_order = np.arange(12)
factors = np.array([30, 15, 5])
# factors = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])

# set the video object
video = Video(video_path)

if video_path == 'video_samples/vibration3.avi':
    video.crop_video(3016)

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
mother_matrix = magnification.video_reconstruction(modal_coordinates_order.size//2, factors)
for result in range(modal_coordinates_order.size//2+1):
    magnification.create_video_from_frames("mode%d" % result, frames=mother_matrix[result])


