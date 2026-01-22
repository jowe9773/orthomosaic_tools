#learning_async.py

"import packages and modules"
from pathlib import Path
import time
import cv2
from functions import File_Functions, Video_Functions, Audio_Functions


#define a orthomosaic video function that will process a single experiment



if __name__ == "__main__":

    #instantiate classes
    ff = File_Functions()
    af = Audio_Functions()
    vf = Video_Functions()

    #define important variables
    COMPRESSION = 2.5
    SPEED = 1
    START_TIME = 0
    LENGTH = 400
    OUT_NAME = "20240708_exp1_long.mp4"


    #define the range of x values that will be used in calculating the error (from 0 to the width of each frame, which in the case of our flume is 2438mm)
    x_range = (0, 2348)


    #get all video files
    main_directory = ff.load_dn("Select a directory with all videos from all of the cameras")
    camera_files = ff.get_sorted_video_filenames(main_directory)
    print(camera_files)

    first_vids = [camera[0] for camera in camera_files if camera]
    
    print(first_vids)

    #get gcps files
    gcps_directory = ff.load_dn("select directory containing gcps files.")
    gcps_filenames = ff.get_gcps_files(gcps_directory)
    print(gcps_filenames)

    targets = []
    for i, gcps in enumerate(gcps_filenames):
        gcps = ff.import_gcps(gcps)
        targets.append(gcps)

    #choose an output location for the final video
    out_vid_dn = ff.load_dn("Select output location for the final video")

    # Start measuring time
    start_time = time.time()

    off_1 = 6600                  #if video from downstream cam is late, add time
    off_2 = 7500                  #if video from downstream cam is early, subtract time
    off_3 = 8500                  #value is in milliseconds

    #Generate time offsets using the first video from each camera
    time_offsets = [0, 0+off_1, 0+off_1+off_2, 0+off_1+off_2+off_3]
    print("Time offsets for video streams:")
    print(time_offsets)

    # Start measuring time
    start_vid_time = time.time()

    #Generate homography matricies
    homo_mats = []
    reprojection_errors = []
    for i, vid in enumerate(first_vids):
        homography = vf.find_homography(i+1, targets[i])
        homo_mats.append(homography)
        error = vf.calculate_reprojection_error(targets[i], homography, x_range = x_range, cam = i)
        reprojection_errors.append(error)

    
    #open captures for each video
    captures_lists = []
    for i, vid_list in enumerate(camera_files):
        captures = []
        for j, file in enumerate(vid_list):   
            cap = cv2.VideoCapture(file)
            captures.append(cap)
        captures_lists.append(captures)

    print(captures_lists)


    vf.orthomosaicing(captures_lists, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION)

    # End measuring time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    video_time = end_time - start_vid_time

    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Time taken to process frames: {video_time:.2f} seconds")
    print("Reproection errors: ", reprojection_errors)
    print("Time offsets: ", time_offsets)
