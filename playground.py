import matplotlib.pyplot as plt
from video_processing import Video
from video_processing import Video_Magnification
import numpy as np
import scipy.io


video_path = 'video_samples/vibration.avi'
if video_path == 'video_samples/vibration2.avi':
    frame_rate = 240
    number_of_frames = 400
else:
    frame_rate = 480
    number_of_frames = 600
# set the video object
video = Video(video_path)
video.set_frames()
# start video magnification
video_magnification = Video_Magnification(video)

# pre-processing
video_magnification.standardize_frames()
time_serie = video_magnification.create_time_series()
mean = np.mean(video_magnification.time_serie, axis=0)
video_magnification.time_serie = video_magnification.time_serie - mean

# Hibert Transform
real_time_serie, imag_time_serie = video_magnification.apply_hilbert_transform()

# dimension reduction
real_vectors, real_values, real_reduced = video_magnification.apply_PCA(real_time_serie)
imag_vectors, imag_values, imag_reduced = video_magnification.apply_PCA(imag_time_serie)

# Sorting stuff
eigen_values = np.append(real_values, imag_values)
real_vectors = real_vectors.T
imag_vectors = imag_vectors.T
eigen_vectors = np.append(real_vectors, imag_vectors, axis=1)
reduced = np.append(real_reduced, imag_reduced, axis=1)
id = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[id]
eigen_vectors = eigen_vectors[:, id]
print(reduced.shape)
# reduced = reduced[:, id]
number_components = 8

# blind source separation
mixture_matrix, unmixed = video_magnification.apply_BSS(reduced[:, 0:number_components].T)
unmixed = np.fliplr(unmixed)
Winvmix = np.flip(np.linalg.inv(mixture_matrix))

# writing
mdic = {"a": reduced, "label": "experiment"}
scipy.io.savemat('components.mat', mdic)

# visualize modes and modal coordinates
mode_shapes = np.matmul(Winvmix, eigen_vectors[:, 0:number_components].T).T
t = np.arange(number_of_frames)/frame_rate
freq = np.arange(number_of_frames)/(number_of_frames/frame_rate)
modal_coordinates = np.array([7, 6, 3, 2, 4, 5, 1, 0])


fig, axs = plt.subplots(len(modal_coordinates), 3)
rows = len(modal_coordinates)
columns = 3
for row in range(rows):
    for column in range(columns):
        axs[row][0].plot(t, unmixed[:, modal_coordinates[row]], color="#069AF3")

        aux = (np.abs(np.fft.fft(unmixed[:, modal_coordinates[row]]))) ** 2
        axs[row][1].plot(freq[2:300], aux[2:300], color="#069AF3")

        aux = (np.angle(np.fft.fft(unmixed[:, modal_coordinates[row]]))) ** 2
        axs[row][2].plot(freq[2:300], aux[2:300], color="#069AF3")

fig2 = plt.figure()
rows = 2
columns = len(modal_coordinates)//2
for row in range(number_components):
        fig2.add_subplot(rows, columns, row+1)
        plt.imshow(mode_shapes[:, row].reshape(video_magnification.frames_heigh,
                                              video_magnification.frames_width), 'gray', aspect='auto')


# video reconstruction
#W = eigen_vectors.T
#mode_shapes = np.dot(principal_components.T, W)
#modal_coordinates = np.matmul(W, dimension_reduced_series.T)
#video_magnification.video_reconstruction(mode_shapes, modal_coordinates, phase_zero)

# plt.plot(modal_coordinates[1, :])
# plt.imshow(mode_shapes[0].reshape(video_magnification.frames_heigh, video_magnification.frames_width), 'gray')
# plt.clf()
