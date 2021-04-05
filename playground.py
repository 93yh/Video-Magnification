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
time_serie = video_magnification.create_time_series()

# Hibert Transform
real_time_serie, imag_time_serie = video_magnification.apply_hilbert_transform()

# dimension reduction
real_dimension_reduced_series, real_principal_components = video_magnification.apply_PCA(real_time_serie)
imag_dimension_reduced_series, imag_principal_components = video_magnification.apply_PCA(imag_time_serie)

# blind source separation
eigen_values, eigen_vectors, principal_components = video_magnification.apply_BSS(real_principal_components, imag_principal_components)

# visualize modes
modes = np.matmul(principal_components.T, eigen_vectors)
dimension_reduced_serie = np.concatenate((real_dimension_reduced_series.T, imag_dimension_reduced_series.T))
modal_coordinates = np.matmul(dimension_reduced_serie.T, eigen_vectors.T)

mode0 = modes[:, 0].reshape(video_magnification.frames_heigh, video_magnification.frames_width).real
plt.imsave('video_samples/mode1.jpeg', mode0, cmap='gray')

mode1 = modes[:, 1].reshape(video_magnification.frames_heigh, video_magnification.frames_width).real
plt.imsave('video_samples/mode2.jpeg', mode1, cmap='gray')

mode2 = modes[:, 2].reshape(video_magnification.frames_heigh, video_magnification.frames_width).real
plt.imsave('video_samples/mode3.jpeg', mode2, cmap='gray')

mode3 = modes[:, 3].reshape(video_magnification.frames_heigh, video_magnification.frames_width).real
plt.imsave('video_samples/mode4.jpeg', mode3, cmap='gray')

mode4 = modes[:, 4].reshape(video_magnification.frames_heigh, video_magnification.frames_width).real
plt.imsave('video_samples/mode5.jpeg', mode4, cmap='gray')

mode5 = modes[:, 5].reshape(video_magnification.frames_heigh, video_magnification.frames_width).real
plt.imsave('video_samples/mode6.jpeg', mode5, cmap='gray')

fig1 = plt.figure()
for j in range(1, 7):
    fig1.add_subplot(2, 3, j)
    plt.plot(modes[:, j-1].real, 'gray')
'''
fig2 = plt.figure()
fig2.subplots_adjust(wspace=0.1)
for j in range(1, 4):
    fig2.add_subplot(2, 4, j)
    plt.imshow(imag_principal_components[j-1].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray', )
fig3 = plt.figure()
for j in range(1, 4):
    fig3.add_subplot(2, 4, j)
    plt.plot(imag_principal_components[j-1], 'gray', )
'''
# video reconstruction
#W = eigen_vectors.T
#mode_shapes = np.dot(principal_components.T, W)
#modal_coordinates = np.matmul(W, dimension_reduced_series.T)
#video_magnification.video_reconstruction(mode_shapes, modal_coordinates, phase_zero)

# plt.plot(modal_coordinates[1, :])
# plt.imshow(mode_shapes[0].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray')
# plt.clf()