#functions.py

'''A file that holds all of the functions used in the process of orthomosaicing the video files.'''

#import neccesary packages and modules
from pathlib import Path
import csv
import tkinter as tk
from tkinter import filedialog
import tempfile
import sys
import moviepy.editor as mp
from scipy.signal import correlate
from scipy.io import wavfile
import cv2
import numpy as np
import asyncio
import concurrent.futures



class File_Functions():
    '''Class containing methods for user handling of files.'''

    def __init__(self):
        print("Initialized File_Funtions")

    def load_dn(self, purpose):
        """this function opens a tkinter GUI for selecting a 
        directory and returns the full path to the directory 
        once selected
        
        'purpose' -- provides expanatory text in the GUI
        that tells the user what directory to select"""

        root = tk.Tk()
        root.withdraw()
        directory_name = filedialog.askdirectory(title = purpose)

        return directory_name

    def load_fn(self, purpose):
        """this function opens a tkinter GUI for selecting a 
        file and returns the full path to the file 
        once selected
        
        'purpose' -- provides expanatory text in the GUI
        that tells the user what file to select"""

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = purpose)

        return filename

    def get_sorted_video_filenames(self, directory):
        # Convert the directory to a Path object
        directory_path = Path(directory)
        
        # Get the list of subdirectories (one for each camera)
        subdirectories = [subdir for subdir in directory_path.iterdir() if subdir.is_dir()]
        
        # Initialize a list to store the lists of filenames for each camera
        camera_files = []
        
        # Iterate through each camera subdirectory
        for subdir in subdirectories:
            # Get all .MP4 files in the subdirectory and store their full paths
            mp4_files = [file for file in subdir.iterdir() if file.suffix == ".MP4"]
            
            # Sort the filenames based on their name
            mp4_files.sort()
            
            # Add the sorted list of full paths to camera_files
            camera_files.append([str(file) for file in mp4_files])

        return camera_files
    
    def get_gcps_files(self, directory):
        # Convert directory to Path object
        directory_path = Path(directory)
        
        # Get all CSV files in the directory and sort them alphabetically
        gcps_files = sorted(directory_path.glob("*.csv"))
        
        # Convert Path objects to strings with consistent forward slashes
        gcps_files = [str(file_path) for file_path in gcps_files]
        
        return gcps_files

    def import_gcps(self, gcps_fn):
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
    
class Audio_Functions():
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor()
        print("Initialized Audio_Functions")

    async def extract_audio_async(self, video_path):
        """Asynchronously extract audio from an MP4 and save it as a temporary WAV file."""
        return await self.loop.run_in_executor(self.executor, self.extract_audio, video_path)

    def extract_audio(self, video_path):
        """This method extracts audio from an MP4 and saves it as a temporary WAV file."""
        print(video_path)
        clip = mp.VideoFileClip(video_path)
        audio = clip.audio
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_path = temp_audio_file.name
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        rate, audio_data = wavfile.read(temp_audio_path)
        clip.reader.close()
        return rate, audio_data

    async def extract_all_audios(self, video_paths):
        """Run extract_audio asynchronously for a list of video paths."""
        tasks = [self.extract_audio_async(video_path) for video_path in video_paths]
        return await asyncio.gather(*tasks)

    def find_time_offset(self, rate1, audio1, rate2, audio2):
        """This function compares two audio files and lines up the wave patterns to match in time.
        Returns the time offset."""

        # Ensure the sample rates are the same
        if rate1 != rate2:
            raise ValueError("Sample rates of the two audio tracks do not match")

        # Convert stereo to mono if necessary
        if len(audio1.shape) == 2:
            audio1 = audio1.mean(axis=1)
        if len(audio2.shape) == 2:
            audio2 = audio2.mean(axis=1)

        # Normalize audio data to avoid overflow
        audio1 = audio1 / np.max(np.abs(audio1))
        audio2 = audio2 / np.max(np.abs(audio2))

        # Compute cross-correlation
        correlation = correlate(audio1, audio2)
        lag = np.argmax(correlation) - len(audio2) + 1

        # Calculate the time offset in milliseconds
        time_offset = lag / rate1 * 1000

        return time_offset

    def find_all_offsets(self, rate_data, audio_data):
        """Find time offsets between the first audio and the others concurrently."""
        tasks = []
        for i in range(1, len(audio_data)):
            task = self.executor.submit(self.find_time_offset, rate_data[i], audio_data[i], rate_data[i-1], audio_data[i-1])
            tasks.append(task)

        time_offsets = [0]
        for i, task in enumerate(tasks):
            offset = task.result() + time_offsets[i]
            time_offsets.append(offset)
            print(f"Offset between video {i+1} and video {i+2} found")

        return time_offsets

class Video_Functions():
    def __init__(self):
        print("Initialized Video_Functions.")

    def find_homography(self, cam, gcps):
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
    
    def calculate_reprojection_error(self, gcps, homography_matrix, x_range=None, cam=0):
        """
        Calculate the reprojection error given ground control points and a homography matrix.
        
        Args:
            gcps (tuple): A tuple containing two sets of points. 
                        gcps[0] - Real-world points (x, y)
                        gcps[1] - Image points (x, y)
            homography_matrix (numpy array): 3x3 Homography matrix for transformation.
            x_range (tuple, optional): A tuple specifying the range of x coordinates (min_x, max_x) to filter the real-world points.
        
        Returns:
            float: Mean reprojection error for the selected points.
        """

        # Convert to float32 numpy arrays
        img_pts = np.array(gcps[1], dtype=np.float32)
        rw_pts = np.array(gcps[0], dtype=np.float32)

        # Add a third dimension to the image points (homogeneous coordinates)
        img_pts = np.column_stack((img_pts, np.ones((img_pts.shape[0], 1))))

        # Apply homography to the source points
        projected_pts = homography_matrix @ img_pts.T

        # Normalize the projected points by dividing by the third coordinate
        projected_pts = projected_pts[:2] / projected_pts[2]

        # Transpose projected points for easier manipulation (back to Nx2)
        projected_pts = projected_pts.T
        
        # If an x_range is provided, filter points based on the real-world x coordinate
        if x_range is not None:
            min_x, max_x = x_range
            # Create a mask based on the x-coordinates of real-world points
            mask = (rw_pts[:, 0] >= min_x) & (rw_pts[:, 0] <= max_x)
            
            # Apply the mask to real-world points
            rw_pts = rw_pts[mask]
            
            # Apply the same mask to the corresponding projected points
            projected_pts = projected_pts[mask]

        # Calculate the Euclidean distance between actual and projected points
        errors = np.linalg.norm(rw_pts - projected_pts, axis=1)

        # Return the average error
        mean_error = np.mean(errors)
        print(f"Reprojection Error for cam {cam+1}: {mean_error}")
        
        return mean_error

    def frame_to_umat_frame(self, frame):
        uframe = cv2.UMat(frame)
        return uframe

    def orthomosaicing(self, captures_list, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION):
        # Describe shape
        final_shape = [2438, 4000]
        compressed_shape = (int(final_shape[0] / COMPRESSION), int(final_shape[1] / COMPRESSION))
        output_shape = (compressed_shape[0] * 4, compressed_shape[1])

        print("out Shape: ", output_shape)

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(out_vid_dn + "/" + OUT_NAME, fourcc, captures_list[0][0].get(cv2.CAP_PROP_FPS) * SPEED, output_shape)

        # Function to process each frame
        def process_frame(cap, homo_mat, black_frame, capture_index, capture_indices):
            # Read the next frame from the video capture
            ret, frame = cap.read()
                  
            # Check if the frame is valid
            if frame is None or frame.size == 0:
                # If we're within the first 5 seconds, return a black frame
                if 0 < cap.get(cv2.CAP_PROP_POS_MSEC) < 5000:
                    print(f"Returning black frame due to missing data.")
                    return black_frame, capture_index  # Return capture index as-is
                
                # Switch to the next capture if we have processed enough frames
                print(f"Warning: Frame is empty or could not be read. Switching to the next capture.")

                # Increment capture index for this camera
                capture_indices[capture_index] += 1  # Adjust based on the current camera index

                # Check if there are more captures available for this camera
                if capture_indices[capture_index] < len(captures_list[capture_index]):
                    # Load the next capture
                    next_capture = captures_list[capture_index][capture_indices[capture_index]]
                    current_caps[capture_index] = next_capture  # Update the global capture list
                    next_ret, next_frame = current_caps[capture_index].read()

                    if next_ret and next_frame is not None:
                        # If the next frame is valid
                        uframe = cv2.UMat(next_frame)  # Convert frame to UMat for processing
                        corrected_frame = cv2.warpPerspective(uframe, homo_mat, final_shape)  # Apply homography
                        corrected_frame = cv2.resize(corrected_frame, compressed_shape)  # Resize frame
                        return corrected_frame.get(), capture_index  # Return the processed frame and updated index
                    else:
                        print(f"No valid frame in the next capture.")
                        return black_frame, capture_index  # If the next frame is also invalid, return black frame
                
                else:
                    print(f"No more captures left for camera {capture_index}.")
                    return black_frame, capture_index  # If no more captures, return black frame

            # If the frame is valid, proceed to process it
            uframe = cv2.UMat(frame)  # Convert frame to UMat for processing
            corrected_frame = cv2.warpPerspective(uframe, homo_mat, final_shape)  # Apply homography
            corrected_frame = cv2.resize(corrected_frame, compressed_shape)  # Resize frame
            return corrected_frame.get(), capture_index  # Return the processed frame and unchanged index

        # Ensure the black frame matches the dimensions and type of the video frames
        def create_black_frame(reference_frame):
            black_frame = np.zeros(reference_frame.shape, dtype=reference_frame.dtype)
            return black_frame

        # Initialize variables for tracking current capture and file
        current_caps = [captures[0] for captures in captures_list]
        capture_indices = [0] * len(captures_list)  # Track which file in each list is being used
        frame_rates = [cap.get(cv2.CAP_PROP_FPS) for cap in current_caps]

        # Get first valid frame to determine dimensions and data type for black frame
        ret, first_frame = current_caps[0].read()
        if not ret or first_frame is None:
            print("Error: Could not read the first frame.")
            sys.exit()
        first_frame_resized = cv2.resize(first_frame, compressed_shape)
        black_frame = create_black_frame(first_frame_resized)  # Black frame now matches other frames

        # Reset start position after reading the first frame
        start_time = START_TIME * 1000
        for i, cap in enumerate(current_caps):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time + time_offsets[i])

        # Initialize processed frames count
        frames_processed = [0] * len(captures_list)

        # Process frames in parallel
        count = 0
        while count <= LENGTH:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a list of arguments for each camera's processing
                args = [(current_caps[i], homo_mats[i], black_frame, i, capture_indices) for i in range(len(current_caps))]
                
                # Process frames concurrently
                results = list(executor.map(lambda p: process_frame(*p), args))

            # Collect corrected frames and update current captures
            corrected_frames = []
            for corrected_frame, index in results:
                corrected_frames.append(corrected_frame)

            # If there are valid frames, merge and write them to the output video
            if corrected_frames:
                merged = cv2.hconcat(corrected_frames)
                out.write(merged)

            count += 1 / frame_rates[0]
            print(f"Processed {count} seconds.")

        # Release all captures and writer objects at the end
        for cap in current_caps:
            if cap:
                cap.release()
        out.release()
        cv2.destroyAllWindows()
