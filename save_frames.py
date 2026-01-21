"""This script will take a set videofiles and extract and save files from them"""

from video_processor import VideoProcessor
from collections import deque
import cv2
from pathlib import Path


if __name__ == "__main__":

    main_path = "C:/Users/jwelsh/Image Annotation/Frames for target annotation/"
    
    videofiles = [
                  main_path + "2.0/GX010134.MP4",
                  main_path + "2.0/GX010176.MP4",
                  main_path + "4.0/GX010165.MP4",
                  main_path + "4.0/GX010182.MP4",
                  main_path + "4.0/GX010225.MP4",
                  main_path + "4.0/GX020225.MP4"]

    out_path = main_path + "Frames"

    for i, videofile in enumerate(videofiles): #iterate through the list of videofiles

        video_name = Path(videofile).stem
        print(video_name)
        
        video = VideoProcessor(videofile) #open the file as a videoprocessor object

        rolling_buffer = deque()
        n = 500 #choose the number of frames between each save
        frames_saved = 0 #start counter of the number of frames saved

        

        for frame, index in video.frames():
            if index % n != 0:
                continue  # skip frames that are not every nth

            # Save the frame as an imagex
            output_path = out_path + "/" + video_name + "_" + str(index) + ".jpg"
            cv2.imwrite(output_path, frame)
            print(f"Frame {index} saved to {output_path}")
            frames_saved += 1
            print(frames_saved)

            if frames_saved >= 32:
                break