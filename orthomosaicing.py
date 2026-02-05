#learning_async.py

"import packages and modules"
from pathlib import Path
import csv
import numpy as np
from collections import deque
from tqdm import tqdm
import cv2



class OrthomosaicTools():
    def __init__(self):
        print("initalized OrthomosaicTools")

    #helper functions
    def _set_gcps(self, directory):
        self.gcps_dir = directory

        # Convert directory to Path object
        directory_path = Path(directory)
        
        # Get all CSV files in the directory and sort them alphabetically
        gcps_files = sorted(directory_path.glob("*.csv"))
        
        # Convert Path objects to strings with consistent forward slashes
        self.gcps_files = [str(file_path) for file_path in gcps_files]

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
        
        self.gcps_by_cam = {} #make an empty list for actual gcps data for each camera to go into

        for i, file in enumerate(self.gcps_files): #for each camera, make a dictionary key:value item with the GCPs
            gcps = import_gcps(file)
            self.gcps_by_cam[f"{i+1}"] = gcps

    def _find_homography(self):
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

        self.matrix_by_cam = {}

        for cam, gcps in self.gcps_by_cam.items():   #for each camera, made a key:value item with the homography matrix
            matrix = find_homography(int(cam), gcps)
            self.matrix_by_cam[f"{cam}"] = matrix

    def _set_start_times(self, timing):
        self.timing = timing
        # NOW SET THE START TIMES FOR EACH CAMERA ENSURING THAT THEY ARE ALL >= 0
        start_time = timing[0]*1000
        start_times_list = [start_time, start_time+timing[2], start_time+timing[2]+timing[3], start_time+timing[2]+timing[3]+timing[4]]
        shift = -min(start_times_list) if min(start_times_list) < 0 else 0
        shifted_start_times = [v + shift for v in start_times_list]

        self.start_times = {}    #put the start times into a dictionary for consistency
        for i, time in enumerate(shifted_start_times):
            self.start_times[f"{i+1}"] = time

    def _set_video_files_by_camera(self, video_dir):

        self.exp_name = video_dir.split("/")[-1]
        self.video_files_by_camera = {}
        for i in range(4):  #for each camera, dive down into the camera directory and grab a list of filepaths for all of the files within, and sort them in alphanumeric order
            camera_dir = Path(video_dir + f"/Cam{i+1}")
            video_files = sorted(camera_dir.iterdir())
            self.video_files_by_camera[f"{i+1}"] = video_files

        self.video_queues = {cam: deque(files) for cam, files in self.video_files_by_camera.items()}  # Prepare deque of video files for each camera

    def _prep_counter(self):
        cap = cv2.VideoCapture(self.video_files_by_camera["1"][0])
        fps = cap.get(cv2.CAP_PROP_FPS) #open a capture to get the frame rate so that we can accurately assess how many frames we need to process to get to the specified length
        cap.release() #release capture to clean up

        self.processed_frame_counter = 0

        self.total_frames = int(fps*self.timing[1]) #set a variable that will be the total number of frames that will be processed
        self.pbar = tqdm(total=self.total_frames, ncols=100, smoothing = 0, colour= "green", desc=f"Processing {self.exp_name}")   
    
    def _create_black_frame(self):
        cap = cv2.VideoCapture(self.video_files_by_camera["1"][0])
        ret, frame = cap.read()
        if frame is not None:
            self.black_frame = np.zeros_like(frame)
        cap.release() #release capture to clean up

    def orthomosaic_experiment(self, gcps_dir, video_dir, timing, output_filepath):
        self._set_gcps(gcps_dir)

        self._find_homography()

        self._set_start_times(timing)

        self._set_video_files_by_camera(video_dir)
        
        self._create_black_frame()

        self._prep_counter()

        # Open first video for each camera
        caps = {}
        fps_by_cam = {}
        for cam, queue in self.video_queues.items():
            if queue:
                cap = cv2.VideoCapture(queue.popleft())
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(fps)
                fps_by_cam[cam] = fps
                caps[cam] = cap
            else:
                caps[cam] = None
                fps_by_cam[cam] = 23.976023976023978  # fallback

        # Set start times according to offsets
        for cam, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_times[cam]/1000*cap.get(cv2.CAP_PROP_FPS)))

        # Track consecutive failures per camera to detect EOF
        temp_fail_counter = {cam: 0 for cam in caps}
        MAX_TEMP_FAILS = 24

        # START A VIDEO WRITER FOR THE OUTPUT
        compression = 2.5
        final_shape = [2438, 4000]
        compressed_shape = (int(final_shape[0] / compression), int(final_shape[1] / compression))
        output_shape = (compressed_shape[0] * 4, compressed_shape[1])


        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_filepath, fourcc, cap.get(cv2.CAP_PROP_FPS), output_shape)

        while self.processed_frame_counter < self.total_frames:
            output_frames = {}  # frames for all cameras at this time step
            
            for cam, cap in caps.items():
                frame_to_use = None

                while True:
                    if cap is None: # No more files for this camera: use black frame
                        frame_to_use = self.black_frame
                        print(f"No more captures for cam{cam}")
                        break
                        

                    ret, frame = cap.read() #read the frame

                    if ret and frame is not None and frame.size > 0: #if frame is valid, reset the temp_fail_counter and use the frame
                        temp_fail_counter[cam] = 0
                        frame_to_use = frame
                        break

                    temp_fail_counter[cam] += 1 #if frame is not valid, then we start the temporary fail counter

                    if temp_fail_counter[cam] >= MAX_TEMP_FAILS: #if we are above the max number of sequential frames that cause problems
                        print("Capture hasnt had a valid frame in 5 seconds, checking for next capture")
                        cap.release()                               #release the capture

                        if self.video_queues[cam]:                                   # if there are more videos in the queue
                            print("Starting nect capture")
                            cap = cv2.VideoCapture(self.video_queues[cam].popleft()) # open the next capture for that cameras
                            caps[cam] = cap
                            temp_fail_counter[cam] = 0                          # reset the fail clock

                        else:
                            # No more chunks: use black frames for rest of output
                            caps[cam] = None
                            frame_to_use = self.black_frame
                            break
                
                output_frames[cam] = frame_to_use
            
            #now that we have the frames, lets correct them
            def process_frames(output_frames):
                corrected_frames = []
                for cam, frame in output_frames.items():
                    uframe = cv2.UMat(frame)  # Convert frame to UMat for processing
                    corrected_frame = cv2.warpPerspective(uframe, self.matrix_by_cam[cam], final_shape)  # Apply homography
                    corrected_frame = cv2.resize(corrected_frame, compressed_shape)  # Resize frame
                    corrected_frames.append(corrected_frame)

                return corrected_frames
            

            corrected_frames_list = process_frames(output_frames)

            # NOW WE CAN STITCH THE FRAMES TOGETHER
            merged = cv2.hconcat(corrected_frames_list)
            out.write(merged)

            self.processed_frame_counter += 1
            self.pbar.update(1)

        self.pbar.close()

        #close captures
        for camera, cap in caps.items():
            if cap is not None:
                cap.release()

        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # CHOOSE DIRECTORIES CONTAINING VIDEO AND GCPS DATA
    video_dir = "G:/video_data/20240603_exp1_goprodata"
    gcps_dir = "C:/Users/jwelsh/Image Annotation/annotated_gcps/20240603"

    # SET AN OUTPUT DIRECTORY
    output_dir = "C:/Users/jwelsh/Image Annotation"
    output_name = "20240603_exp1_test.mp4"
    output_filepath = output_dir + "/" + output_name

    # MAKE A LIST CONTAINING ALL TIMING INFO (Start Time [s], Length [s], Offset 1 [ms], Offset 2 [ms], Offset 3 [ms])
    timing = [690, 60, 9750, 6350, 7500]

    ortho = OrthomosaicTools()

    ortho.orthomosaic_experiment(gcps_dir= gcps_dir, video_dir= video_dir, timing= timing, output_filepath=output_filepath)