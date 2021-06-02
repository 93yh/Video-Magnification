import cv2
import numpy as np


class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.number_of_frames = None
        self.frames = self.set_frames()
        self.frames_shape = self.frames[0].shape[0: 2]
        self.number_of_pixels = self.frames_shape[0] * self.frames_shape[1]
        self.fps = self.define_fps()
        self.gray_frames = self.produce_gray_frames()

    def set_frames(self):
        print("Setting video frames")
        video = cv2.VideoCapture(self.video_path)
        self.number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("The video contains %d frames, reading...\n" % self.number_of_frames)
        frames = None
        for read in range(self.number_of_frames):
            flag, frame = video.read()
            if read == 0:
                frames = np.zeros((self.number_of_frames, frame.shape[0], frame.shape[1], 3), 'uint8')
            if (cv2.waitKey(1) and 0xFF == ord('q')) or flag is False:
                break
            frames[read] = frame
        video.release()
        return frames

    def define_fps(self):
        print("Defining video FPS")
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        print('FPS of the video: ', fps, '\n')
        return fps

    def produce_gray_frames(self):
        print('produzing gray frames\n')
        gray_frames = np.zeros((self.number_of_frames, self.frames_shape[0], self.frames_shape[1]))
        for frame in range(self.number_of_frames):
            gray_frames[frame] = cv2.cvtColor(self.frames[frame], cv2.COLOR_BGR2GRAY)
        return gray_frames

    def visualize_video(self):
        video = cv2.VideoCapture(self.video_path)
        while True:
            flag, frame = video.read()
            if (cv2.waitKey(1) and 0xFF == ord('q')) or flag is False:
                break
            cv2.imshow('Video', frame)
        video.release()
        cv2.destroyAllWindows()

    def crop_video(self, frame):
        print("Cropping video\n")
        # This funtion cuts out all the frames that come before the input frame
        self.frames = self.frames[frame:]
        self.gray_frames = self.gray_frames[frame:]
        self.number_of_frames = self.frames.shape[0]

    def rotate_video_90_degrees(self):
        print("rotating video\n")
        gray_frames_shape = self.gray_frames[0].shape
        new_gray_frames = np.zeros((self.number_of_frames, gray_frames_shape[1], gray_frames_shape[0]))
        for frame in range(0, self.number_of_frames):
            frame_gray_to_rotate = np.copy(self.gray_frames[frame])
            for rotate in range(3):
                frame_gray_to_rotate = np.rot90(frame_gray_to_rotate)
            new_gray_frames[frame] = frame_gray_to_rotate
        self.gray_frames = new_gray_frames
        self.frames_shape = self.gray_frames[0].shape[0: 2]

    def downsample_frames(self, factor):
        shape = self.gray_frames[0][0::factor, 0::factor].shape
        new_gray_frames = np.zeros((self.number_of_frames, shape[0], shape[1]))
        for frame in range(0, self.number_of_frames):
            new_gray_frames[frame] = self.gray_frames[frame][0::factor, 0::factor]
        self.gray_frames = new_gray_frames
        self.number_of_pixels = self.gray_frames[0].shape[0] * self.gray_frames[0].shape[1]
        self.frames_shape = self.gray_frames[0].shape[0: factor]
