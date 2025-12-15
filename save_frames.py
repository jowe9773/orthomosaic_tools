"""This script will take a set videofiles and extract and save files from them"""

from video_processor import VideoProcessor
from collections import deque
import cv2


if __name__ == "__main__":
    videofiles = ["C:/Users/jwelsh/Image Annotation/20240529_Video_to_Frames/20240529_exp2_1.mp4"]

    out_path = "C:/Users/jwelsh/Image Annotation/20240529_Video_to_Frames/Frames"

    for i, videofile in enumerate(videofiles): #iterate through the list of videofiles

        video = VideoProcessor(videofile) #open the file as a videoprocessor object

        rolling_buffer = deque()
        n = 24 #choose the number of frames between each save
        frames_saved = 0 #start counter of the number of frames saved

        for frame, index in video.frames():
            print(index)
            if index % n != 0:
                continue  # skip frames that are not every nth

            # Save the frame as an image
            output_path = out_path + "/" + "20240529_exp2_1" + "_" + str(index) + ".jpg"
            cv2.imwrite(output_path, frame)
            print(f"Frame {index} saved to {output_path}")

        video.cap.release()
        cv2.destroyAllWindows()