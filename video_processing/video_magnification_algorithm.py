import cv2
import numpy as np
from video_processing.complexity_pursuit_functions import return_mask
from scipy.signal import lfilter
from scipy import linalg
from scipy.signal import hilbert
from video_processing.pca_functions import apply_pca
from video_processing.visualization_functions import plot_components_or_sources
from video_processing.visualization_functions import plot_mode_shapes_and_modal_coordinates
import matplotlib.pyplot as plt
import random


class Video_Magnification:
    def __init__(self, video):
        self.video = video
        self.time_serie = None
        self.time_serie_mean = None
        self.real_time_serie = None
        self.imag_time_serie = None
        self.real_vectors = None
        self.imag_vectors = None
        self.real_values = None
        self.imag_values = None
        self.real_components = None
        self.imag_components = None
        self.eigen_vectors = None
        self.eigen_values = None
        self.components = None
        self.sources = None
        self.mixture_matrix = None
        self.mode_shapes = None
        self.modal_coordinates = None
        self.error = None
        self.norm = None
        self.reconstructed = None
        self.encryption_key = None

    def create_time_series(self):
        print('Creating time series\n')
        self.time_serie = np.zeros((self.video.number_of_frames, self.video.number_of_pixels))
        for frame in range(self.video.number_of_frames):
            vector = self.video.gray_frames[frame].ravel()
            self.time_serie[frame] = vector
        return self.time_serie

    def remove_background(self):
        self.time_serie_mean = np.mean(self.time_serie, axis=0)
        self.time_serie = self.time_serie - self.time_serie_mean

    def scramble(self):
        print("Scrambling the pixels of the video\n")
        permutation_vector = random.sample(range(self.video.number_of_pixels), self.video.number_of_pixels)
        self.time_serie = self.time_serie[:, permutation_vector]
        self.encryption_key = np.arange(self.video.number_of_pixels)
        indexes = np.argsort(permutation_vector)
        self.encryption_key = self.encryption_key[indexes]
        return self.encryption_key

    def apply_hilbert_transform(self):
        print("Applying Hilbert Transform in the time series\n")
        hilbert_data = hilbert(self.time_serie, axis=0)
        hilbert_data = hilbert(hilbert_data.imag, axis=1)
        real_time_serie = np.copy(self.time_serie)
        imag_time_serie = np.imag(hilbert_data)
        self.real_time_serie = real_time_serie
        self.imag_time_serie = imag_time_serie
        return self.real_time_serie, self.imag_time_serie

    def dimension_reduction(self):
        print('Apllying PCA in the phase series')
        real_vectors, real_values, real_components = apply_pca(self.real_time_serie)
        imag_vectors, imag_values, imag_components = apply_pca(self.imag_time_serie)
        # sorting
        eigen_values = np.append(real_values, imag_values)
        real_vectors = real_vectors.T
        imag_vectors = imag_vectors.T
        eigen_vectors = np.append(real_vectors, imag_vectors, axis=1)
        components = np.append(real_components, imag_components, axis=1)
        id = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[id]
        eigen_vectors = eigen_vectors[:, id]
        components = components[:, id]
        self.eigen_vectors = eigen_vectors
        self.eigen_values = eigen_values
        self.components = components
        return self.eigen_vectors, self.eigen_values, self.components

    def extract_sources(self, number_components):
        components = self.components[:, 0:number_components]
        print('Applying BSS')
        short_mask = return_mask(1.0, 10, 50)
        long_mask = return_mask(900000.0, 10, 50)
        print('calculating filters')
        short_filter = lfilter(short_mask, 1, components, axis=0)
        long_filter = lfilter(long_mask, 1, components, axis=0)
        print('Calculating covariance matrix')
        short_cov = np.cov(short_filter, rowvar=False)
        long_cov = np.cov(long_filter, rowvar=False)
        print('Calculating eigenvectors and eigenvalues')
        eigen_values, mixture_matrix = linalg.eig(long_cov, short_cov)
        print('mixing matrix shape: ', mixture_matrix.shape, '\n')
        unmixed = -np.matmul(components, mixture_matrix)
        unmixed = -np.flip(unmixed, axis=1)
        self.sources = unmixed
        self.mixture_matrix = mixture_matrix
        return self.mixture_matrix, self.sources

    def create_mode_shapes_and_modal_coordinates(self, number_components, order):
        print("Creating mode shapes and modal coordinates")
        winvmix = np.flip(np.linalg.inv(self.mixture_matrix), axis=0)
        mode_shapes = np.matmul(winvmix, self.eigen_vectors[:, 0:number_components].T).T
        modal_coordinates = self.sources[:, order]
        mode_shapes = mode_shapes[:, order]
        self.mode_shapes = mode_shapes
        self.modal_coordinates = modal_coordinates
        print("Size of mode shapes in bytes: ", self.mode_shapes.nbytes)
        print("Size of modal coordinates in bytes: ", self.modal_coordinates.nbytes, '\n')
        return self.mode_shapes, self.modal_coordinates

    def visualize_components_or_sources(self, subject, order):
        if subject == "components":
            print("Visualizing components\n")
            visualize = self.components
        elif subject == "sources":
            print("visualizing sources\n")
            visualize = self.sources
        elif subject == "modal coordinates":
            print("visualizing modal coordinates\n")
            visualize = self.modal_coordinates
        else:
            print("Subject is wrong, inform if you want to see components or sources")
            return None
        t = np.arange(self.video.number_of_frames) / self.video.fps
        freq = np.arange(self.video.number_of_frames) / (self.video.number_of_frames / self.video.fps)
        rows = len(order)
        columns = 3
        plot_components_or_sources(rows, columns, t, freq, visualize, order)

    def visualize_mode_shapes_and_modal_coordinates(self, order, do_unscramble=False):
        t = np.arange(self.video.number_of_frames) / self.video.fps
        columns = len(order)
        plot_mode_shapes_and_modal_coordinates(self, columns, t, do_unscramble)

    def video_reconstruction(self, factor_1=30, factor_2=15, factor_3=5, do_unscramble=False):
        print("Reconstruting video from mode shapes and modal coordinates\n")
        frames_0 = np.zeros((self.video.number_of_frames, self.video.frames_shape[0], self.video.frames_shape[1]))
        frames_1 = np.zeros((self.video.number_of_frames, self.video.frames_shape[0], self.video.frames_shape[1]))
        frames_2 = np.zeros((self.video.number_of_frames, self.video.frames_shape[0], self.video.frames_shape[1]))
        frames_3 = np.zeros((self.video.number_of_frames, self.video.frames_shape[0], self.video.frames_shape[1]))
        source0 = np.matmul(self.modal_coordinates, self.mode_shapes.T)
        first_part = np.matmul(self.modal_coordinates[:, [0, 1]], self.mode_shapes[:, [0, 1]].T)
        second_part = np.matmul(self.modal_coordinates[:, [2, 3]], self.mode_shapes[:, [2, 3]].T)
        third_part = np.matmul(self.modal_coordinates[:, [4, 5]], self.mode_shapes[:, [4, 5]].T)
        source1 = (factor_1 * first_part) - second_part - third_part
        source2 = -first_part + (factor_2 * second_part) - third_part
        source3 = -first_part - second_part + (factor_3 * third_part)
        if do_unscramble:
            background = self.time_serie_mean[self.encryption_key].reshape(self.video.frames_shape)
        else:
            background = self.time_serie_mean.reshape(self.video.frames_shape)
        self.error = np.zeros(self.video.frames_shape, dtype='int64')
        for row in range(self.video.number_of_frames):
            if not do_unscramble:
                frame_0 = source0[row, :].reshape(self.video.frames_shape) + background
                frame_1 = source1[row, :].reshape(self.video.frames_shape) + background
                frame_2 = source2[row, :].reshape(self.video.frames_shape) + background
                frame_3 = source3[row, :].reshape(self.video.frames_shape) + background
            else:
                frame_0 = source0[row, self.encryption_key].reshape(self.video.frames_shape) + background
                frame_1 = source1[row, self.encryption_key].reshape(self.video.frames_shape) + background
                frame_2 = source2[row, self.encryption_key].reshape(self.video.frames_shape) + background
                frame_3 = source3[row, self.encryption_key].reshape(self.video.frames_shape) + background
            frames_0[row] = frame_0
            frames_1[row] = frame_1
            frames_2[row] = frame_2
            frames_3[row] = frame_3

        self.reconstructed = np.copy(frames_0)
        frames_0 = ((frames_0 - frames_0.min()) * (1 / (frames_0.max() - frames_0.min()) * 255)).astype('uint8')
        frames_1 = ((frames_1 - frames_1.min()) * (1 / (frames_1.max() - frames_1.min()) * 255)).astype('uint8')
        frames_2 = ((frames_2 - frames_2.min()) * (1 / (frames_2.max() - frames_2.min()) * 255)).astype('uint8')
        frames_3 = ((frames_3 - frames_3.min()) * (1 / (frames_3.max() - frames_3.min()) * 255)).astype('uint8')
        return frames_0, frames_1, frames_2, frames_3

    def calculate_error(self):
        print('Calculating error and norm betwen original and reconstructed videos')
        self.error = np.zeros(self.video.frames_shape, dtype='float64')
        for frame in range(self.video.number_of_frames):
            self.error = self.error + (self.video.gray_frames[frame].astype('float64') - self.reconstructed[frame])
        self.norm = np.sum(self.error.ravel())**2
        print("Final Norm: ", self.norm, '\n')
        return self.error, self.norm

    def create_video_from_frames(self, name, frames=None, fps=None):
        if frames is None:
            frames = self.video.frames
        if fps is None:
            fps = self.video.fps
        print('Creating video from the frames\n')
        height, width = self.video.frames_shape
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('video_samples/%s.avi' % name, fourcc, fps, size, 0)
        for i in range(len(frames)):
            out.write(frames[i])
        out.release()
