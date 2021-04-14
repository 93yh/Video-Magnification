import cv2
import numpy as np
from video_processing.complexity_pursuit_functions import return_mask
from scipy.signal import lfilter
from scipy import linalg
from scipy.signal import hilbert
from video_processing.pca_functions import apply_pca
from video_processing.visualization_functions import plot_components_or_sources
from video_processing.visualization_functions import plot_mode_shapes_and_modal_coordinates


class Video_Magnification:
    def __init__(self, video):
        self.video = video
        self.time_serie = None
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

    def create_time_series(self):
        print('Creating time series\n')
        pixels_number = self.video.frames_shape[0] * self.video.frames_shape[1]
        self.time_serie = np.zeros((self.video.number_of_frames, pixels_number))
        for frame in range(self.video.number_of_frames):
            vector = self.video.gray_frames[frame].ravel()
            self.time_serie[frame] = vector
        mean = np.mean(self.time_serie, axis=0)
        self.time_serie = self.time_serie - mean
        return self.time_serie

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
        print('Apllying PCA in the phase series\n')
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
        return eigen_vectors, eigen_values, components

    def extract_sources(self, number_components):
        components = self.components[:, 0:number_components]
        print('Applying BSS')
        short_mask = return_mask(1.0, 10, 50)
        long_mask = return_mask(900000.0, 10, 50)
        print('Calculando filtros')
        short_filter = lfilter(short_mask, 1,  components, axis=0)
        long_filter = lfilter(long_mask, 1, components, axis=0)
        print('Calculando matrizes de covariância')
        short_cov = np.cov(short_filter, rowvar=False)
        long_cov = np.cov(long_filter, rowvar=False)
        print('Calculando Auto Valores e Auto Vetores')
        eigen_values, mixture_matrix = linalg.eig(long_cov, short_cov)
        print('shape da matriz de mistura: ', mixture_matrix.shape, '\n')
        unmixed = -np.matmul(components, mixture_matrix)
        unmixed = -np.flip(unmixed, axis=1)
        self.sources = unmixed
        self.mixture_matrix = mixture_matrix
        return self.mixture_matrix, self.sources

    def create_mode_shapes_and_modal_coordinates(self, number_components, order):
        winvmix = np.flip(np.linalg.inv(self.mixture_matrix), axis=0)
        mode_shapes = np.matmul(winvmix, self.eigen_vectors[:, 0:number_components].T).T
        modal_coordinates = self.sources[:, order]
        mode_shapes = mode_shapes[:, order]
        self.mode_shapes = mode_shapes
        self.modal_coordinates = modal_coordinates
        return mode_shapes, modal_coordinates

    def visualize_components_or_sources(self, subject, order):
        if subject == "components":
            print("Visualizing components\n")
            visualize = self.components
        elif subject == "sources":
            print("visualizing sources\n")
            visualize = self.sources
        else:
            print("Subject is wrong, inform if you want to see components or sources")
            return None
        t = np.arange(self.video.number_of_frames) / self.video.fps
        freq = np.arange(self.video.number_of_frames) / (self.video.number_of_frames / self.video.fps)
        rows = len(order)
        columns = 3
        plot_components_or_sources(rows, columns, t, freq, visualize, order)

    def visualize_mode_shapes_and_modal_coordinates(self, order):
        t = np.arange(self.video.number_of_frames) / self.video.fps
        columns = len(order)
        plot_mode_shapes_and_modal_coordinates(self, columns, t)
    '''
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
    '''