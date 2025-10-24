"""This script will take a set videofiles and extract and save files from them"""

from video_processor import VideoProcessor
from collections import deque
import cv2


if __name__ == "__main__":
    videofiles = []

    out_path = ""

    for i, videofile in enumerate(videofiles): #iterate through the list of videofiles

        video = VideoProcessor(videofile) #open the file as a videoprocessor object

        rolling_buffer = deque()
        n =720 #choose the number of frames between each save (24 fps take 1 every 30 seconds)
        frames_saved = 0 #start counter of the number of frames saved

        for frame, index in video.frames():
            if index % n != 0:
                continue  # skip frames that are not every nth

            # Save the frame as an image
            output_path = out_path + "/" + videofile + "_" + index + ".jpg"
            cv2.imwrite(output_path, frame)
            print(f"Frame {index} saved to {output_path}")

            video.view(frame)

            if frames_saved >= 26:
                break


        video.cap.release()
        cv2.destroyAllWindows()