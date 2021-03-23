import cv2
import numpy as np
import matplotlib.pyplot as plt


class Video_Magnification:
    def __init__(self, video):
        self.video = video
        self.filtered_frames = []

    def standardize_frames(self):
        for frame in range(len(self.video.frames)):
            self.video.frames[frame] = cv2.cvtColor(self.video.frames[frame], cv2.COLOR_BGR2GRAY)

    def create_filter(self, ):
        rows, cols = self.video.frames[0].shape
        center_row, center_col = int(rows / 2), int(cols / 2)
        low_filter_mask = np.zeros((rows, cols, 2), np.uint8)
        high_filter_mask = np.ones((rows, cols, 2), np.uint8)
        low_filter_radius = 100
        high_filter_radius = 30
        center = [center_row, center_col]
        x, y = np.ogrid[: rows, :cols]
        low_filter_mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= low_filter_radius * low_filter_radius
        low_filter_mask[low_filter_mask_area] = 1
        high_filter_mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= high_filter_radius * high_filter_radius
        high_filter_mask[high_filter_mask_area] = 0
        return high_filter_mask * low_filter_mask

    def apply_filter(self):
        for frame in range(len(self.video.frames)):
            frame_fft = cv2.dft(np.float32(self.video.frames[frame]), flags=cv2.DFT_COMPLEX_OUTPUT)
            frame_shifted = np.fft.fftshift(frame_fft)
            filter = self.create_filter()
            filtered_frame = frame_shifted * filter
            filter = 20 * np.log(cv2.magnitude(filtered_frame[:, :, 0], filtered_frame[:, :, 1]))
            filter = np.asanyarray(filter, dtype=np.uint8)
            self.filtered_frames.append(filter)
            reverse_shift = np.fft.ifftshift(filtered_frame)
            reversed_image = cv2.idft(reverse_shift)
            reversed_image_magnitude = 20 * np.log(cv2.magnitude(reversed_image[:, :, 0], reversed_image[:, :, 1]))
            result = cv2.normalize(reversed_image_magnitude, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            self.video.frames[frame] = result

    def remove_background(self):
        pass

    def apply_PCA(self):
        pass

    def apply_BSS(self):
        pass

    def create_video_from_frames(self):
        height, width = self.video.frames[0].shape
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('frames.avi', fourcc, 20.0, size, 0)
        out2 = cv2.VideoWriter('filter.avi', fourcc, 20.0, size, 0)
        for i in range(len(self.video.frames)):
            out.write(self.video.frames[i])
            out2.write(self.filtered_frames[i])
        out.release()
