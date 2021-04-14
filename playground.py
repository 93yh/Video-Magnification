from video_processing import Video
from video_processing import Video_Magnification
import numpy as np
import scipy.io


video_path = 'video_samples/vibration.avi'
number_components = 8
components_order = np.arange(number_components)
sources_order = np.arange(number_components)
modal_coordinates_order = np.array([7, 6, 0, 1, 5, 4])

# set the video object
video = Video(video_path)

# Start video magnification
video_magnification = Video_Magnification(video)

# Create time series
time_serie = video_magnification.create_time_series()

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

# visualize modal coordinates and mode shapes
video_magnification.visualize_mode_shapes_and_modal_coordinates(modal_coordinates_order)

# video reconstruction
#W = eigen_vectors.T
#mode_shapes = np.dot(principal_components.T, W)
#modal_coordinates = np.matmul(W, dimension_reduced_series.T)
#video_magnification.video_reconstruction(mode_shapes, modal_coordinates, phase_zero)

# plt.plot(modal_coordinates[1, :])
# plt.imshow(mode_shapes[0].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray')
# plt.clf()
