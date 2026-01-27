# This is a way of getting a functioning software - before I make it into a
# function based situation. I need to work through the big picture so I can organize the code in a way that makes sense.

# Load neccesary packages and modules
import pandas as pd
from pathlib import Path
import cv2
from pprint import pprint
import csv
import numpy as np

# Identify the directories containing the data that you need
video_dir = "G:/video_data" #directory with all video data
gcps_dir = "C:/Users/jwelsh/Image Annotation/annotated_gcps" # directory with gcps files for all experiments
time_info_file = "F:/Videos/time_offsets_v1.csv"
out_dir = "C:/Users/jwelsh/Image Annotation/"

# Initialize variables that you need to organize the data
time_info = pd.read_csv(time_info_file) # load time info file into a pandas database
exp_name_list = time_info["exp_name"].tolist() # extract a list of all of the experiments to process

# Iterate through each experiment
for i, exp in enumerate(exp_name_list):
    print(f"Now working on processing {exp}")
    exp_date = exp.split("_")[0]

    # Get timing details for the experiment
    times_list = time_info.loc[time_info["exp_name"] == exp, ["Start Time", "Length", "Offset 1", "Offset 2", "Offset 3"]].iloc[0].tolist() 

    # Make a dictionary where keys are cameras (there will be 4 key:item pairs) and each item is a list containing all of the video file names for that camera
    video_exp_dir = video_dir + "/" + exp + "_goprodata" #get the directory name by combining the video_dir_with the experiment name and naming scheme
    video_files_dict = {} #make empty dictionary for video files by camera

    for i in range(4):
        camera = f"Cam{i+1}" #name the camera that is currently being grabbed

        cam_dir = Path(video_exp_dir + "/" + camera) #name the subdirectory for the camera
        files = sorted(cam_dir.iterdir()) #extract the file names in this directory
        
        video_files_dict[camera] = files # save those files in the dictionary of cameras and their files


    # Make a dictionary where keys are cameras (there will be 4 key:item pairs) and each item is the gcps file for that camera
    gcps_exp_dir = Path(gcps_dir + "/" + exp_date) #get the directory name by combining the gcps_dir with the experiment date
    gcps_files_dict = {} #make empty dictionary for video files by camera

    for i in range(4):
        camera = f"Cam{i+1}" #name the camera that is currently being grabbed
        file = sorted(p for p in gcps_exp_dir.iterdir() if p.is_file() and camera in p.name)[0]
        gcps_files_dict[camera] = file # save those files in the dictionary of cameras and their files

    # Now Import the GCPs from the files
    def import_gcps(gcps_fn):
        """module for importing ground control points as lists"""

        gcps_rw_list = [] #make list for real world coordinates of GCPs
        gcps_image_list = [] #make list for image coordinates of GCPs

        #Read csv file into a list of real world and a list of image gcp coordinates
        with open(gcps_fn, 'r', newline='') as csvfile:
            # Create a CSV reader object
            csv_reader = csv.reader(csvfile)

            # Skip the header row
            next(csv_reader)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Each row is a list where each element represents a column value
                gcps_image_list.append(row[1:3])
                gcps_rw_list.append(row[3:5])

                gcps = [gcps_rw_list, gcps_image_list]

        return gcps
    
    
    gcps = {}
    for i in range(4):
        gcps_file = gcps_files_dict[f"Cam{i+1}"]
        gcps[f"Cam{i+1}"] = import_gcps(gcps_file)

    pprint(gcps)


    """Okay, now we have gathered all of the necceary information to start so its time to actually process the video!"""

    # First set up an output file name
    out_vid_name = out_dir + "/" + exp + "_test.mp4"

    # Next, define the final size of each frame (this is based on the size of the flume and how big you want the pixels to be (compression 1 means each pixel is 1mm square)
    width = 9750
    height = 4000
    compression = 2.5
    final_size = (width/2.5, height/2.5)

    # Now we can find the homography matrix for each of the cameras
    def find_homography(cam, gcps):
        """Method for finding homography matrix."""

        #adjust the ground control points so that they are within the frame of the camera, which starts at (0,0) for each camera
        for count, i in enumerate(gcps[0]):
            i[0] = float(i[0]) - 2438 * (cam-1)
            i[1] = (float(i[1])*-1) + 2000

        #convert the image and destination coordinates to numpy array with float32
        src_pts = np.array(gcps[1])
        src_pts = np.float32(src_pts[:, np.newaxis, :])

        dst_pts = np.array(gcps[0])
        dst_pts = np.float32(dst_pts[:, np.newaxis, :])

        #now we can find homography matrix
        h_matrix = cv2.findHomography(src_pts, dst_pts)

        return h_matrix[0]

    homographies = {}
    for i, camera in enumerate(gcps):
        homographies[camera] = find_homography(i+1, gcps[camera])

    #Now, take the reported offsets and prepare the start times for each camera
    start_times = [times_list[0], 
                   times_list[0] + times_list[2]/1000,
                   times_list[0] + (times_list[2] + times_list[3])/1000, 
                   times_list[0] + (times_list[2] + times_list[3] + times_list[4])/1000]

    # Shift the start times so that they are all >= 0
    shift = -min(start_times) if min(start_times) < 0 else 0
    shifted_values = [v + shift for v in start_times]
    print(shifted_values)

    # Set the time on each cameras capture accordingly
    cap1 = cv2.VideoCapture(video_files_dict["Cam1"][0])
    cap1.set(cv2.CAP_PROP_POS_MSEC, shifted_values[0])

    cap2 = cv2.VideoCapture(video_files_dict["Cam2"][0])
    cap2.set(cv2.CAP_PROP_POS_MSEC, shifted_values[1])

    cap3 = cv2.VideoCapture(video_files_dict["Cam3"][0])
    cap3.set(cv2.CAP_PROP_POS_MSEC, shifted_values[2])

    cap4 = cv2.VideoCapture(video_files_dict["Cam4"][0])
    cap4.set(cv2.CAP_PROP_POS_MSEC, shifted_values[3])

    # set up frame counters for (1) each capture and (2) overall
    fcount1 = 0
    fcount2 = 0
    fcount3 = 0
    fcount4 = 0

    fcount_overall = 0

    #OKAY its finally actually time to process the video frames. This has two parts: (1) orthorectifying and cropping the cameras and (2) stitching them together
    #first, lets write a function that will process each frame

    def process_frame(cap, frame_counter):
        ret, frame = cap.read() #read the current frame

        frame_counter += 1

        print(ret)
        print(frame_counter)

    process_frame(cap1, fcount1)
        


    

    # Release all captures and writer objects at the end

    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()