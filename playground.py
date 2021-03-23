import cv2
from video_processing import Video
from video_processing import Video_Magnification


video_path = 'video_samples/vibration.mp4'

# set the video object
video = Video(video_path)
video.set_frames()

# start video magnification
video_magnification = Video_Magnification(video)
video_magnification.standardize_frames()
video_magnification.apply_filter()
video_magnification.create_video_from_frames()
