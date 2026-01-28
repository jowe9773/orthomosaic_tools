#learning_async.py

"import packages and modules"
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import time
import cv2
from pprint import pprint

# CHOOSE DIRECTORIES CONTAINING VIDEO AND GCPS DATA
video_dir = "G:/video_data/20240627_exp1_goprodata"
gcps_dir = "C:/Users/jwelsh/Image Annotation/annotated_gcps/20240627"

# SET AN OUTPUT DIRECTORY
output_dir = "C:/Users/jwelsh/Image Annotation"
output_name = "20240627_exp1_text.mp4"
output_filepath = output_dir + "/" + output_name

# MAKE A LIST CONTAINING ALL TIMING INFO (Start Time [s], Length [s], Offset 1 [ms], Offset 2 [ms], Offset 3 [ms])
timing = [0, 1, 0, 0, 0]

# NOW EXTRACT GCPS DATA
def get_gcps_files_from_dir(directory):
    # Convert directory to Path object
    directory_path = Path(directory)
    
    # Get all CSV files in the directory and sort them alphabetically
    gcps_files = sorted(directory_path.glob("*.csv"))
    
    # Convert Path objects to strings with consistent forward slashes
    gcps_files = [str(file_path) for file_path in gcps_files]
    
    return gcps_files

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

gcps_files_list = get_gcps_files_from_dir(gcps_dir) #get a list of the filepaths for each camera and put them in alphanumerical order

gcps_by_cam = {} #make an empty list for actual gcps data for each camera to go into

for i, file in enumerate(gcps_files_list): #for each camera, make a dictionary key:value item with the GCPs
    gcps = import_gcps(file)
    gcps_by_cam[f"{i+1}"] = gcps

# NOW ITS TIME TO FIND HOMOGRAPHY FOR EACH CAMERA
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

matrix_by_cam = {}

for cam, gcps in gcps_by_cam.items():   #for each camera, made a key:value item with the homography matrix
    matrix = find_homography(int(cam), gcps)
    matrix_by_cam[f"{cam}"] = matrix

# NOW SETUP THE VIDEOS INTO A DICTIONARY OF LISTS FOR EACH CAMERA
video_files_by_camera = {}
for i in range(4):  #for each camera, dive down into the camera directory and grab a list of filepaths for all of the files within, and sort them in alphanumeric order
    camera_dir = Path(video_dir + f"/Cam{i+1}")
    video_files = sorted(camera_dir.iterdir())
    video_files_by_camera[f"{i+1}"] = video_files

# NOW SET THE START TIMES FOR EACH CAMERA ENSURING THAT THEY ARE ALL >= 0
start_time = timing[0]*1000
start_times_list = [start_time, start_time+timing[2], start_time+timing[2]+timing[3], start_time+timing[2]+timing[3]+timing[4]]
shift = -min(start_times_list) if min(start_times_list) < 0 else 0
shifted_start_times = [v + shift for v in start_times_list]

start_times = {}    #put the start times into a dictionary for consistency
for i, time in enumerate(shifted_start_times):
    start_times[f"{i+1}"] = time

# GET A COUNTER READY FOR KEEPING TRACK OF DURATION OF FINAL VIDEO AND ENDING IT WHEN IT REACHES THE LENGTH SPECIFIED BY USER   
processed_frame_counter = 0

cap = cv2.VideoCapture(video_files_by_camera["1"][0])
fps = cap.get(cv2.CAP_PROP_FPS) #open a capture to get the frame rate so that we can accurately assess how many frames we need to process to get to the specified length
cap.release() #release capture to clean up

total_frames = int(fps*timing[1]) #set a variable that will be the total number of frames that will be processed


# NOW SET UP THE CAPTURES TO ITERATE THROUGH THE VIDEO AND COUNTERS TO KEEP TRACK OF WHAT FRAME WE ARE ON FOR EACH CAMERA
caps = {}

for camera, files in video_files_by_camera.items():

    cap = cv2.VideoCapture(files[1]) #open video file

    start_time = start_times[camera] #set start frame based on start time
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time/1000*fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #apply start frame to the capture

    caps[f"{camera}"] = cap #add capture to dictionary 


# FINALLY, BEFORE WE BEGIN TO ITERATE THROUGH THE FRAMES, WE NEED TO MAKE AN EQUIVALENT BLACK FRAME TO USE IN CASES WHERE ITS APPROPRIATE
black_frame = None

# NOW ITERATE THROUGH FRAMES AND ORTHOCROP THEN STITCH TOGETHER
while processed_frame_counter <= total_frames:
    for camera, cap in caps.items():
        print("Cam", camera)
        
        ret, frame = cap.read()

        if ret == True: #this means that we are still within the capture, and there are frames to read
            if frame is not None and frame.size >0: #if the frame is valid and has data in it to read, then we orthocrop it
                print(frame.size)
                print("This is a good frame and we are going to use it!")
                continue

            else: #if the frame is not valid, we will replace it with a black frame to act as a placeholder
                print("This frame is no good, so we are going to use a black frame as a placeholder")
                continue

        if ret == False: #this means that we are beyond the capture, and we need to switch to the next one or not have a single one
            print(ret)
            print("The capture is over and we need to switch to the next video file")
            continue

        else:
            print(ret)
            continue

    
    processed_frame_counter += 1
    print(processed_frame_counter)


#close captures
for camera, cap in caps.items():
    cap.release()
