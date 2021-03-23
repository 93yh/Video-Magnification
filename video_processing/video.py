import cv2


class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []
        self.phase_serie = None
        self.amplitude_serie = None

    def set_frames(self):
        video = cv2.VideoCapture(self.video_path)
        while True:
            flag, frame = video.read()
            if (cv2.waitKey(1) and 0xFF == ord('q')) or flag is False:
                break
            self.frames.append(frame)
        video.release()

    def visualize_video(self):
        video = cv2.VideoCapture(self.video_path)
        while True:
            flag, frame = video.read()
            if (cv2.waitKey(1) and 0xFF == ord('q')) or flag is False:
                break
            cv2.imshow('Video', frame)
        video.release()
        cv2.destroyAllWindows()
