#learning_async.py

"import packages and modules"
import time
import asyncio
import cv2
from functions import File_Functions, Video_Functions, Audio_Functions


if __name__ == "__main__":

    #instantiate classes
    ff = File_Functions()
    af = Audio_Functions()
    vf = Video_Functions()

    #define important variables
    COMPRESSION = 4
    SPEED = 1
    START_TIME = 0
    LENGTH = 708
    OUT_NAME = "test_full.mp4"

    #load video files:
    vid1 = ff.load_fn("Select video from camera 1")
    vid2 = ff.load_fn("Select video from camera 2")
    vid3 = ff.load_fn("Select video from camera 3")
    vid4 = ff.load_fn("Select video from camera 4")

    videos = [vid1, vid2, vid3, vid4]

    #load gcps files:
    gcps1 = ff.load_fn("Select gcps file for camera 1")
    gcps2 = ff.load_fn("Select gcps file for camera 2")
    gcps3 = ff.load_fn("Select gcps file for camera 3")
    gcps4 = ff.load_fn("Select gcps file for camera 4")

    gcpss = [gcps1, gcps2, gcps3, gcps4]

    targets = []
    for i, gcps in enumerate(gcpss):
        gcps = ff.import_gcps(gcps)
        targets.append(gcps)
    
    print(targets)

    #choose an output location for the final video
    out_vid_dn = ff.load_dn("Select output location for the final video")

    # Start measuring time
    start_time = time.time()

    #Generate time offsets
    rate_data = []
    audio_data = []
    loop = asyncio.get_event_loop()
    rates_and_audios = loop.run_until_complete(af.extract_all_audios(videos))

    for i, tup in enumerate(rates_and_audios):
        rate_data.append(tup[0])
        audio_data.append(tup[1])

    print("FPS for video streams:")
    print(rate_data)

    #find time offsets
    time_offsets = af.find_all_offsets(rate_data, audio_data)

    print("Time offsets for video streams:")
    print(time_offsets)

    # Start measuring time
    start_vid_time = time.time()

    #Generate homography matricies
    homo_mats = []
    for i, vid in enumerate(videos):
        homography = vf.find_homography(i+1, targets[i])
        homo_mats.append(homography)
        vf.calculate_reprojection_error(targets, homography)

    #Open capture for each video stream
    captures = []
    for i, vid in enumerate(videos):
        cap = cv2.VideoCapture(vid)
        captures.append(cap)

    vf.orthomosaicing(captures, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION)

    # End measuring time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    video_time = end_time - start_vid_time

    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Time taken to process frames: {video_time:.2f} seconds")
