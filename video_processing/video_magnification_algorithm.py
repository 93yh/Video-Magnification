import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from video_processing.complexity_pursuit_functions import return_mask
from scipy.signal import lfilter
from scipy import linalg
from scipy.signal import hilbert



class Video_Magnification:
    def __init__(self, video):
        self.video = video
        self.frames_heigh= video.frames[0].shape[0]
        self.frames_width = video.frames[0].shape[1]
        self.time_serie = None
        self.filter_shape = []

    def standardize_frames(self):
        print('Standardizing frames of the video\n')
        for frame in range(len(self.video.frames)):
            self.video.frames[frame] = cv2.cvtColor(self.video.frames[frame], cv2.COLOR_RGB2GRAY)

    def create_time_series(self):
        print('Creating time series\n')
        pixels_number = self.video.frames[0].shape[0] * self.video.frames[0].shape[1]
        self.time_serie = np.zeros((len(self.video.frames), pixels_number))
        for frame in range(len(self.video.frames)):
            vector = self.video.frames[frame].ravel()
            self.time_serie[frame] = vector
        return self.time_serie

    def apply_hilbert_transform(self):
        print("Applying Hilbert Transform in the time series\n")
        hilbert_data = hilbert(self.time_serie, axis=0)
        hilbert_data = hilbert(hilbert_data.imag, axis=1)
        real_time_serie = np.copy(self.time_serie)
        imag_time_serie = np.imag(hilbert_data)
        return real_time_serie, imag_time_serie

    def apply_PCA(self, time_serie):
        print('Apllying PCA in the phase series\n')
        pca = PCA()
        reduced = pca.fit_transform(time_serie)
        print("Shape da matriz reduzida: ", reduced.shape)
        print("Shape dos auto vetores: ", pca.components_.shape, '\n')
        return pca.components_, pca.singular_values_, reduced

    def apply_PCA2(self, time_serie):
        print('Apllying PCA in the phase series')
        pca = PCA()
        eigen_vectors, eigen_values, Vt = linalg.svd(time_serie.T, full_matrices=False)
        reduced = np.matmul(eigen_vectors.T, time_serie.T)
        print("Shape da matriz reduzida: ", reduced.shape)
        print("Shape dos auto vetores: ", eigen_vectors.shape, '\n')
        return eigen_vectors, eigen_values, reduced

    def apply_BSS(self, principal_components):
        print('Applying BSS')
        short_mask = return_mask(1.0, 10, 50)
        long_mask = return_mask(900000.0, 10, 50)
        print('Calculando filtros')
        short_filter = lfilter(short_mask, 1,  principal_components, axis=0)
        long_filter = lfilter(long_mask, 1, principal_components, axis=0)
        print('Calculando matrizes de covariância')
        short_cov = np.cov(short_filter, rowvar=False)
        long_cov = np.cov(long_filter, rowvar=False)
        print('Calculando Auto Valores e Auto Vetores')
        eigen_values, mixture_matrix = linalg.eig(long_cov, short_cov)
        print('shape da matriz de mistura: ', mixture_matrix.shape, '\n')
        unmixed = -np.matmul(principal_components, mixture_matrix)
        return mixture_matrix, unmixed

    def apply_BSS2(self, principal_components):
        print('Applying BSS')
        ica = FastICA(max_iter=1000)
        nsei = ica.fit_transform(principal_components.T)
        mixture_matrix = ica.components_
        print(nsei.shape)
        print('shape da matriz de mistura: ', mixture_matrix.shape, '\n')
        unmixed = np.matmul(principal_components.T, mixture_matrix)
        return mixture_matrix, nsei

    def video_reconstruction(self, mode_shapes, modal_coordinates, phase_zero):
        matrix = np.matmul(modal_coordinates.T, mode_shapes.T)
        matrix = np.insert(matrix, 0, phase_zero, axis=0)
        result = self.video.amplitude_serie * np.exp(1j * matrix)
        frame = result[1].reshape(self.frames_heigh, self.frames_width)
        fft = np.fft.fft2(frame)
        ifft = np.fft.ifft2(fft)
        plt.imshow(ifft.imag, 'gray')
        print(result)

    def create_video_from_frames(self, name, frames=None):
            if not frames:
                frames = self.video.frames
            print('Creating video from the frames\n')
            height, width = self.video.frames[0].shape
            size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('video_samples/%s' % name, fourcc, 20.0, size, 0)
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()
