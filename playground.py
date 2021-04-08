import matplotlib.pyplot as plt
from video_processing import Video
from video_processing import Video_Magnification
import numpy as np


video_path = 'video_samples/vibration2.avi'

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
real_vectors, real_values, real_reduced = video_magnification.apply_PCA(real_time_serie)
imag_vectors, imag_values, imag_reduced = video_magnification.apply_PCA(imag_time_serie)

# Sorting stuff
eigen_values = np.append(real_values, imag_values)
eigen_vectors = np.append(real_vectors, imag_vectors, axis=1)
reduced = np.append(real_reduced, imag_reduced, axis=1)
id = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[id]
eigen_vectors = eigen_vectors[:, id]
reduced = reduced[:, id]
number_components = 6

# blind source separation
mixture_matrix, unmixed = video_magnification.apply_BSS(reduced[:, 0:number_components].T)
unmixed = -1 * np.fliplr(unmixed)
'''
# visualize modes
modes = eigen_vectors

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
'''
t = np.arange(400)/240
fig1 = plt.figure()
for j in range(1, 7):
    fig1.add_subplot(2, 3, j)
    plt.plot(t, unmixed[:, j-1])

fig2 = plt.figure()
fig2.subplots_adjust(wspace=0.1)
for j in range(1, 7):
    fig2.add_subplot(2, 4, j)
    plt.imshow(eigen_vectors[:, j-1].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray', )

fig3 = plt.figure()
freq = np.arange(400)/(400/240)
for j in range(1, 7):
    aux = np.abs(np.fft.fft(unmixed[:, j-1])) ** 2
    fig3.add_subplot(2, 3, j)
    plt.plot(freq[2:300], aux[2:300])

# video reconstruction
#W = eigen_vectors.T
#mode_shapes = np.dot(principal_components.T, W)
#modal_coordinates = np.matmul(W, dimension_reduced_series.T)
#video_magnification.video_reconstruction(mode_shapes, modal_coordinates, phase_zero)

# plt.plot(modal_coordinates[1, :])
# plt.imshow(mode_shapes[0].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray')
# plt.clf()
