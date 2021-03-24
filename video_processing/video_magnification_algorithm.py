import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Video_Magnification:
    def __init__(self, video):
        self.video = video
        self.magnitude_frames = []
        self.frames_heigh= video.frames[0].shape[0]
        self.frames_width = video.frames[0].shape[1]
        self.filter_shape = []

    def standardize_frames(self):
        print('Standardizing frames of the video\n')
        for frame in range(len(self.video.frames)):
            self.video.frames[frame] = cv2.cvtColor(self.video.frames[frame], cv2.COLOR_BGR2GRAY)

    def create_filter(self, ):
        print('Creating filter for the frames\n')
        rows, cols = self.video.frames[0].shape
        center_row, center_col = int(rows / 2), int(cols / 2)
        low_filter_mask = np.zeros((rows, cols), np.uint8)
        high_filter_mask = np.ones((rows, cols), np.uint8)
        low_filter_radius = 180
        high_filter_radius = 60
        center = [center_row, center_col]
        x, y = np.ogrid[: rows, :cols]
        low_filter_mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= low_filter_radius * low_filter_radius
        low_filter_mask[low_filter_mask_area] = 1
        high_filter_mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= high_filter_radius * high_filter_radius
        high_filter_mask[high_filter_mask_area] = 0
        plt.imsave('filter.jpeg', high_filter_mask * low_filter_mask, cmap='gray')
        return high_filter_mask * low_filter_mask

    def apply_filter(self):
        print('Filter process initiated\n')
        for frame in range(len(self.video.frames)):
            frame_fft = np.fft.fft2(self.video.frames[frame])
            frame_shifted = np.fft.fftshift(frame_fft)
            frame_magnitude = 20 * np.log(np.abs(frame_shifted))
            frame_magnitude = cv2.normalize(frame_magnitude, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            self.magnitude_frames.append(frame_magnitude)
            filter = self.create_filter()
            filtered_frame = frame_shifted * filter
            reverse_shift = np.fft.ifftshift(filtered_frame)
            reversed_image = np.fft.ifft2(reverse_shift)
            reversed_image = np.abs(reversed_image)
            result = cv2.normalize(reversed_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            self.video.frames[frame] = result

    def create_time_series(self):
        print('Creating time series\n')
        pixels_number = self.video.frames[0].shape[0] * self.video.frames[0].shape[1]
        self.video.phase_serie = np.zeros((len(self.video.frames), pixels_number))
        self.video.amplitude_serie = np.zeros((len(self.video.frames), pixels_number))
        for frame in range(len(self.video.frames)):
            fft_frame = np.fft.fft2(self.video.frames[frame])
            frame_shifted = np.fft.fftshift(fft_frame)
            frame_phase = np.angle(frame_shifted)
            frame_amplitude = 20 * np.log(np.abs(frame_shifted))
            phase_vector = frame_phase.ravel()
            amplitude_vector = frame_amplitude.ravel()
            self.video.phase_serie[frame] = phase_vector
            self.video.amplitude_serie[frame] = amplitude_vector

    def remove_background(self):
        print("removing background of the video using the time series\n")
        self.video.phase_serie = self.video.phase_serie[1:, :]
        amplitude_mean = np.mean(self.video.amplitude_serie, axis=0)
        amplitude_mean = np.uint8(np.abs(amplitude_mean))
        flag, otsu = cv2.threshold(amplitude_mean, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        columns_deleted = []
        for pixel in range(len(otsu)):
            if otsu[pixel] <= 0:
                columns_deleted.append(pixel)
        self.video.phase_serie = np.delete(self.video.phase_serie, columns_deleted, 1)

    def remove_background2(self):
        print("removing background of the video using the time series\n")
        self.video.phase_serie = self.video.phase_serie[1:, :]
        amplitude_mean = np.mean(self.video.amplitude_serie, axis=0)
        amplitude_mean = np.uint8(np.abs(amplitude_mean))
        flag, otsu = cv2.threshold(amplitude_mean, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        columns_deleted = []
        height = self.video.phase_serie.shape[0]
        for pixel in range(len(otsu)):
            if otsu[pixel] <= 0:
                columns_deleted.append(pixel)
        for column in columns_deleted:
            self.video.phase_serie[:, column] = np.full(height, 255, np.uint8)
        frame = self.video.phase_serie[0, :].reshape(self.frames_heigh, self.frames_width)
        plt.imsave('phase_pos_pre_processing.jpeg', frame, cmap='gray')

    def apply_PCA(self):
        print('Apllying PCA in the phase series')
        pca = PCA()
        dimension_reduced_series = pca.fit_transform(self.video.phase_serie)
        print(dimension_reduced_series)
        print(dimension_reduced_series.shape)
        print(pca.components_[0].shape)
        return dimension_reduced_series, pca.components_

    def apply_BSS(self):
        pass

    def create_video_from_frames(self, name, frames=None):
        if not frames:
            frames = self.video.frames
        print('Creating video from the frames\n')
        height, width = self.video.frames[0].shape
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('%s' % name, fourcc, 20.0, size, 0)
        for i in range(len(frames)):
            out.write(frames[i])
        out.release()
